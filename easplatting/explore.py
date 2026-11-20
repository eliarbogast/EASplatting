"""explore.py — search spaces a 1-D importance sweep cannot cover (EA_PLAN next step 1 and "more ideas").

Genome = [feature weights (k, optional)] + [one importance cutoff per region (R)].
  score    = sum_i w_i * rank_i          rank_i = percentile rank of per-Gaussian feature i (opacity, volume, max, ...)
  keep     = rank(score) >= cutoff[region]   (global rank, or the rank inside the region with --local)
With weights fixed to "max only" and one region this IS the 1-D importance sweep, so every setting here nests the baseline.

    python -m easplatting.explore combo --scene bonsai --features max,opacity,volume,views,camdist --regions shell4
    python -m easplatting.explore fps --scene bonsai
Every `combo` run appends one JSON line per (method, seed) to results/explore/results.jsonl. Setting: 32 fitness views, mean >= 40 dB,
worst view >= 35 dB, `max` importance, the same as the main experiment (docs/EA_PLAN.md), so the numbers compare with figures/README.md.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from easplatting.cmaes import cma_es
from easplatting.ea import evolutionary_algorithm
from easplatting.experiment import Log, importance_scores, setup_scene
from easplatting.gs_loader import prune
from easplatting.metrics import psnr_fast
from easplatting.render import gaussian_contributions, measure_fps, render_views, select_indices

FEATURES = ["opacity", "volume", "max", "sum", "density", "views", "camdist", "lg", "fisher", "learned", "fisherj", "learned2",
            "maxscale", "aniso", "tailq", "tailmin", "edge", "foot", "depth", "knn", "maxloo", "sumloo"]
CACHED = ("fisherj", "learned2", "tailq", "tailmin", "edge", "foot", "depth", "knn", "maxloo", "sumloo")  # computed by `scores`
DEAD = 0.05


# ------------------------------------------------------------ features, regions, masks (no rendering)


def rank(x, average_ties=False):
    """Percentile rank in [0, 1). Ties broken by index, or (average_ties) all given their mean rank."""
    if average_ties:
        _, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
        start = torch.cumsum(counts, 0) - counts
        return (start + (counts - 1) / 2).float()[inverse] / len(x)
    r = torch.empty(len(x), device=x.device)
    r[x.argsort()] = torch.arange(len(x), device=x.device, dtype=torch.float32)
    return r / len(x)


TIE_AVERAGED = ("tailq", "tailmin")  # more than half of the Gaussians share one value: index order would be noise


def feature_rank(name, values):
    return rank(values, average_ties=name in TIE_AVERAGED)


def camera_distance(means, cams, chunk=500_000):
    """Distance from each Gaussian to the nearest train camera centre (the 'distance to the camera hull' feature)."""
    centres = cams.centers().to(means.device)
    return torch.cat([torch.cdist(means[i:i + chunk], centres).amin(dim=1) for i in range(0, len(means), chunk)])


def raw_feature(name, ctx, scene_name):
    scene, cams, train_idx = ctx["scene"], ctx["cams"], ctx["train_idx"]
    if name == "opacity":
        return scene["opacities"]
    if name == "volume":
        return torch.log(scene["scales"].clamp(min=1e-12)).sum(-1)
    if name == "camdist":
        return camera_distance(scene["means"], cams.subset(train_idx))
    if name == "views":
        path = Path("results/cache") / f"{scene_name}_feat_views.pt"
        if not path.exists():
            hits = gaussian_contributions(scene, cams.subset(train_idx).scaled(0.5), batch_size=4)["visible_views"]
            torch.save(hits.cpu(), path)
        return torch.load(path, map_location="cuda").float()
    if name == "maxscale":
        return scene["scales"].amax(-1)
    if name == "aniso":
        return scene["scales"].amax(-1) / scene["scales"].amin(-1).clamp(min=1e-12)
    if name in CACHED:
        return torch.load(Path("results/cache") / f"{scene_name}_importance_{name}.pt", map_location="cuda")
    return importance_scores(scene_name, scene, cams, train_idx, name).cuda()  # sum, max, density, lg, fisher, learned


def feature_ranks(names, ctx, scene_name):
    return torch.stack([feature_rank(n, raw_feature(n, ctx, scene_name).float()) for n in names], dim=1)


def region_ids(kind, ctx, scene_name):
    """(N,) region index per Gaussian and the region count. none | shellR (camera-distance quantile shells) | cell8 (PCA octants)."""
    n = len(ctx["scene"]["means"])
    if kind == "none":
        return torch.zeros(n, dtype=torch.long, device="cuda"), 1
    if kind.startswith("shell"):
        r = int(kind[5:])
        return (rank(raw_feature("camdist", ctx, scene_name)) * r).long().clamp(max=r - 1), r
    if kind == "cell8":
        m = ctx["scene"]["means"]
        centred = m - m.median(dim=0).values
        _, _, v = torch.linalg.svd(centred[:: max(1, n // 200_000)], full_matrices=False)
        proj = centred @ v.T
        bits = (proj > proj.median(dim=0).values).long()
        return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2], 8
    raise ValueError(kind)


def local_rank(score, regions, count):
    """Percentile rank of the score inside its own region."""
    out = torch.empty_like(score)
    for r in range(count):
        idx = (regions == r).nonzero()[:, 0]
        out[idx] = rank(score[idx])
    return out


def make_mask_fn(feats, regions, count, learn_weights, fixed_feature=None, local=False):
    """feats (N, k) ranks. Returns mask_fn(genome) -> bool mask, and the genome length."""
    k = feats.shape[1]
    fixed_rank = feats[:, fixed_feature] if fixed_feature is not None else None
    cached = {}

    def mask_fn(genome):
        g = np.asarray(genome, dtype=float)
        if learn_weights:
            w = torch.tensor(2 * g[:k] - 1, dtype=torch.float32, device=feats.device)
            score = feats @ w
            r = local_rank(score, regions, count) if local else rank(score)
            cut = g[k:]
        else:
            if "r" not in cached:  # weights fixed: rank once
                cached["r"] = local_rank(fixed_rank, regions, count) if local else fixed_rank
            r, cut = cached["r"], g
        cut = torch.tensor(np.where(cut <= DEAD, 0.0, cut), dtype=torch.float32, device=feats.device)
        return r >= cut[regions]

    return mask_fn, (k if learn_weights else 0) + count


def warm_genome(k, count, learn_weights, fixed_feature, cutoff):
    g = np.full(k, 0.5)
    g[fixed_feature] = 1.0
    return np.concatenate([g, np.full(count, cutoff)]) if learn_weights else np.full(count, cutoff)


# ------------------------------------------------------------------------------------ fitness


def make_fitness(scene, mask_fn, views, baselines, tau, worst_tau, prefilter=0, margin=4.0):
    """prefilter K > 0: multi-fidelity. Render K of the fitness views first; if their mean PSNR is more than `margin` dB under tau,
    skip the full render and report that shortfall (still ranks infeasible genomes). fitness.counts tells how many were skipped."""
    sub = list(range(0, len(views), max(1, len(views) // prefilter))) if prefilter else None
    counts = dict(full=0, skipped=0)

    def fitness(genome):
        mask = mask_fn(genome)
        kept = mask.sum().item() / mask.numel()
        if mask.sum() < 1000:
            return -999999.0, kept, 0.0
        pruned = prune(scene, mask)
        if sub is not None:
            p = psnr_fast(render_views(pruned, views.subset(sub)), baselines[sub], max_db=60).mean().item()
            if p < tau - margin:
                counts["skipped"] += 1
                return -1000.0 - (tau - p), kept, p
        counts["full"] += 1
        v = psnr_fast(render_views(pruned, views), baselines, max_db=60)
        psnr = v.mean().item()
        shortfall = max(tau - psnr, worst_tau - v.min().item())
        return (1.0 - kept, kept, psnr) if shortfall <= 0 else (-1000.0 - shortfall, kept, psnr)

    fitness.counts = counts
    return fitness


def sweep_cutoff(fit, points=64, refine=12):
    """The 1-D baseline: one global cutoff on the fixed feature, a sweep then bisection (the main experiment's refined sweep)."""
    for c in np.linspace(0, 1, points):
        fit(np.array([c]))
    ok = [r for r in fit.rows if r["fitness"] >= 0]
    if not ok:
        return None
    low = max(ok, key=lambda r: r["fitness"])["genome"][0]
    high = min(low + 1 / (points - 1), 1)
    for _ in range(refine):
        mid = (low + high) / 2
        if fit(np.array([mid]))[0] >= 0:
            low = mid
        else:
            high = mid
    return low


def diagnose(scene, mask, ctx):
    """Held-out numbers for a mask: all train views (half res) and test views (full res), vs the unpruned renders."""
    out = {}
    for name, idx, scale in (("train", ctx["train_idx"], 0.5), ("test", ctx["test_idx"], 1.0)):
        v = ctx["cams"].subset(idx).scaled(scale)
        base = ctx["bases"].setdefault(name, render_views(scene, v, batch_size=4))
        p = psnr_fast(render_views(prune(scene, mask), v, batch_size=4), base, max_db=60)
        out[f"{name}_psnr"], out[f"{name}_worst"] = p.mean().item(), p.min().item()
    return out


# ------------------------------------------------------------------------------ combo command


def cmd_combo(a):
    np.random.seed(0)
    ctx = setup_scene(a.scene, 4, "max")
    ctx["bases"] = {}
    scene, cams = ctx["scene"], ctx["cams"]
    names = a.features.split(",")
    feats = feature_ranks(names, ctx, a.scene)
    regions, count = region_ids(a.regions, ctx, a.scene)
    views = cams.subset(select_indices(len(cams), "train", k=a.views)).scaled(a.scale)
    baselines = render_views(scene, views)
    fixed = names.index(a.fixed)

    # 1-D baseline in this very setup (fixed max-only weights, one global cutoff)
    base_fn, _ = make_mask_fn(feats, torch.zeros_like(regions), 1, False, fixed)
    base_fit = Log(make_fitness(scene, base_fn, views, baselines, a.tau, a.worst_tau), "sweep", 0)
    cutoff = sweep_cutoff(base_fit)
    base_kept = min((r["kept"] for r in base_fit.rows if r["fitness"] >= 0), default=None)
    if base_kept is None:
        print(f"[{a.name}] {a.scene} 1-D sweep on {a.fixed}: no feasible cutoff", flush=True)
        base_kept = 1.0
    else:
        print(f"[{a.name}] {a.scene} 1-D sweep on {a.fixed}: {100 * base_kept:.1f}% kept at cutoff {cutoff:.4f}", flush=True)

    mask_fn, n = make_mask_fn(feats, regions, count, a.learn_weights, fixed, a.local)
    fit_fn = make_fitness(scene, mask_fn, views, baselines, a.tau, a.worst_tau, a.prefilter)
    warm = warm_genome(len(names), count, a.learn_weights, fixed, cutoff)
    # a warm start is only valid if the warm genome reproduces the baseline (sanity check, one evaluation)
    wf = fit_fn(warm)
    print(f"[{a.name}] genes={n}  warm genome -> kept {100 * wf[1]:.1f}% psnr {wf[2]:.2f} feasible={wf[0] >= 0}", flush=True)

    with open(Path("results/explore") / "results.jsonl", "a") as f:
        f.write(json.dumps(dict(name=a.name, scene=a.scene, method="sweep", seed=0, features=a.fixed, regions="none",
                                feasible=base_kept is not None, kept=base_kept, sweep_kept=base_kept, cutoff=cutoff)) + "\n")
    budget_gens = (a.evals - 32) // 64
    out = Path("results/explore")
    out.mkdir(parents=True, exist_ok=True)
    for method in filter(None, a.methods.split(",")):
        for seed in range(a.seeds):
            np.random.seed(seed)
            fit = Log(fit_fn, method, seed)
            t0 = time.perf_counter()
            if method == "ga":
                init = [np.clip(warm + np.random.randn(n) * a.warm_noise * (i > 0), 0, 1) for i in range(32)] if a.warm else None
                evolutionary_algorithm(fit, budget_gens, 32, 64, n, a.mutation, init_genomes=init)
            elif method == "cmaes":
                cma_es(fit, (32 + 64 * budget_gens) // 32, 32, n, a.warm_noise * 2 if a.warm else 0.3,
                       init_mean=warm if a.warm else None)
            elif method == "random":
                for _ in range(32 + 64 * budget_gens):
                    fit(np.random.rand(n))
            ok = [r for r in fit.rows if r["fitness"] >= 0]
            best = max(fit.rows, key=lambda r: r["fitness"])
            row = dict(name=a.name, scene=a.scene, method=method, seed=seed, features=a.features, regions=a.regions,
                       local=a.local, learn_weights=a.learn_weights, warm=a.warm, genes=n, evals=len(fit.rows),
                       seconds=round(time.perf_counter() - t0, 1), feasible=bool(ok), kept=best["kept"], fit_psnr=best["psnr"],
                       sweep_kept=base_kept, genome=best["genome"], prefilter=a.prefilter,
                       ms_per_eval=float(np.mean([r["ms"] for r in fit.rows])), skipped=fit_fn.counts["skipped"])
            fit_fn.counts.update(full=0, skipped=0)
            if ok:
                row.update(diagnose(scene, mask_fn(np.array(best["genome"])), ctx))
                # evaluations until the first feasible genome that beats the 1-D baseline
                row["evals_to_beat_sweep"] = next((i for i, r in enumerate(fit.rows) if r["fitness"] >= 0 and r["kept"] < base_kept), None)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{a.name}] {a.scene} {method:>6} s{seed}: kept {100 * row['kept']:5.1f}%  (sweep {100 * base_kept:.1f}%)"
                  + (f"  test {row['test_psnr']:.2f} worst {row['test_worst']:.2f}  beat-sweep@{row['evals_to_beat_sweep']}" if ok else "  INFEASIBLE"),
                  flush=True)


# --------------------------------------------------------------------------------- fps command


def cmd_fps(a):
    """FPS of pruned scenes (EA_PLAN next step 3): the importance-sweep optimum, and a few fixed kept fractions, vs the unpruned scene."""
    ctx = setup_scene(a.scene, 4, "max")
    scene, cams = ctx["scene"], ctx["cams"]
    test = cams.subset(ctx["test_idx"])
    stats = ctx["stats"]
    rows = []
    opacity_cut = stats["opacity"].sort().values[len(stats["opacity"]) // 2]  # keeps the top half by opacity
    cases = [("unpruned", torch.ones_like(stats["importance"], dtype=torch.bool))]
    cases += [(f"importance cutoff {c:.2f}", stats["importance"] >= c) for c in (0.2, 0.35, 0.5, 0.65)]
    cases += [("opacity top 50%", stats["opacity"] >= opacity_cut)]
    for label, mask in cases:
        sub = prune(scene, mask)
        r = measure_fps(sub, test, n_warmup=10, n_iters=60)
        rows.append(dict(scene=a.scene, label=label, kept=mask.float().mean().item(), **r))
        print(f"{a.scene:>8} {label:>24}: kept {100 * rows[-1]['kept']:5.1f}%  {r['ms_per_frame']:6.2f} ms  {r['fps']:7.1f} fps", flush=True)
    with open("results/explore/fps.jsonl", "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def cmd_scores(a):
    """Compute and cache a gradient-based importance score (EA_PLAN next step 2)."""
    from easplatting.render import render_views as _render
    from easplatting.scores import fisher_jacobian, learned_gates

    ctx = setup_scene(a.scene, 4, "max")
    t0 = time.perf_counter()
    if a.mode in ("viewfeats", "knn", "loo"):
        if a.mode == "viewfeats":  # tailq, tailmin, edge, foot, depth: one pass over all train views
            from easplatting.features import view_features

            out = view_features(ctx["scene"], ctx["cams"].subset(ctx["train_idx"]).scaled(0.5))
        elif a.mode == "knn":
            from easplatting.features import neighbor_counts

            out = dict(knn=neighbor_counts(ctx["scene"]["means"], ctx["scene"]["scales"]))
        else:  # maxloo, sumloo: importance from the train views that the fitness does NOT use
            from easplatting.render import per_view_contributions

            keep = sorted(set(ctx["train_idx"]) - set(select_indices(len(ctx["cams"]), "train", k=a.fitness_views)))
            c = per_view_contributions(ctx["scene"], ctx["cams"].subset(keep).scaled(0.5))
            out = dict(maxloo=c["max_weight"], sumloo=c["blend_weight"])
        for name, score in out.items():
            torch.save(score.cpu(), Path("results/cache") / f"{a.scene}_importance_{name}.pt")
        print(f"{a.scene}: {a.mode} -> {', '.join(out)} in {time.perf_counter() - t0:.0f}s", flush=True)
        return
    views = ctx["cams"].subset(ctx["train_idx"]).scaled(0.5)
    views = views.subset(np.linspace(0, len(views) - 1, min(a.views, len(views))).astype(int))
    if a.mode == "fisherj":
        score = fisher_jacobian(ctx["scene"], views, probes=a.probes)
    else:  # learned2: the gate learner with normalized gradients, more steps
        score = learned_gates(ctx["scene"], views, _render(ctx["scene"], views), steps=a.steps, normalize=True)
    path = Path("results/cache") / f"{a.scene}_importance_{a.mode}.pt"
    torch.save(score.cpu(), path)
    print(f"{a.scene}: {a.mode} over {len(views)} views x {a.probes} probes in {time.perf_counter() - t0:.0f}s -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combo")
    c.add_argument("--scene", default="bonsai")
    c.add_argument("--name", default="combo")
    c.add_argument("--features", default="max", help="comma list of " + ",".join(FEATURES) + "; must include max")
    c.add_argument("--fixed", default="max", help="feature used when weights are not learned (and by the 1-D baseline)")
    c.add_argument("--learn-weights", action="store_true", help="evolve a weight per feature (else score = max only)")
    c.add_argument("--regions", default="none", help="none | shell<R> | cell8: one cutoff gene per region")
    c.add_argument("--local", action="store_true", help="cutoffs are fractions removed inside each region")
    c.add_argument("--warm", action="store_true", help="start GA / CMA-ES at the 1-D sweep optimum")
    c.add_argument("--warm-noise", type=float, default=0.05)
    c.add_argument("--methods", default="ga,cmaes,random")
    c.add_argument("--seeds", type=int, default=3)
    c.add_argument("--evals", type=int, default=1312)
    c.add_argument("--mutation", type=float, default=0.1)
    c.add_argument("--views", type=int, default=32)
    c.add_argument("--scale", type=float, default=0.5, help="fitness view resolution as a fraction of full (0.5 = main experiment)")
    c.add_argument("--prefilter", type=int, default=0, help="multi-fidelity: screen on this many of the fitness views first")
    c.add_argument("--tau", type=float, default=40.0)
    c.add_argument("--worst-tau", type=float, default=35.0)
    sc = sub.add_parser("scores")
    sc.add_argument("--scene", default="bonsai")
    sc.add_argument("--views", type=int, default=48)
    sc.add_argument("--probes", type=int, default=2)
    sc.add_argument("--mode", default="fisherj", choices=["fisherj", "learned2", "viewfeats", "knn", "loo"])
    sc.add_argument("--fitness-views", type=int, default=32, help="loo: the fitness view count whose views are left out")
    sc.add_argument("--steps", type=int, default=600)
    f = sub.add_parser("fps")
    f.add_argument("--scene", default="bonsai")
    a = ap.parse_args()
    {"combo": cmd_combo, "fps": cmd_fps, "scores": cmd_scores}[a.cmd](a)


if __name__ == "__main__":
    main()
