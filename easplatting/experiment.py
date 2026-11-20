"""experiment.py — GA vs baselines on one scene, one JSONL row per fitness evaluation.

    python -m easplatting.experiment --scene bonsai --genes 4 --seeds 3

Methods (all minimise kept fraction subject to PSNR >= tau vs the unpruned render, same fitness function):
  ga           evolutionary_algorithm (ea.py), parents + children x generations evaluations
  random       the same number of uniform random genomes
  cmaes        CMA-ES (cmaes.py) with the same evaluation budget
  sweep_<gene> 1-D sweep of one gene with every other gene off (deterministic, run once)
Output: results/<scene>_<genes>g[_v<views>][_w<floor>][_res<scale>]/log.jsonl (every evaluation) and summary.json (best per run,
held-out PSNRs), then plot_run.publish() copies plots.png, summary.json and renders.jpg into the tracked figures/runs/<run>/ and refreshes
figures/compare.png: commit those with every run.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from easplatting.cameras import load_cameras, load_image
from easplatting.cmaes import cma_es
from easplatting.ea import evolutionary_algorithm
from easplatting.fitness import make_fitness_function
from easplatting.genome import decode
from easplatting.gs_loader import activate, load_scene, prune
from easplatting.metrics import psnr_fast
from easplatting.render import (
    coverage_indices,
    gaussian_contributions,
    per_view_contributions,
    render_views,
    select_indices,
)
from easplatting.thresholds import _quantile, gene_stats, keep_mask, slider_bounds

GENES = ["opacity", "scale", "anisotropy", "importance"]
OFF = [0.0, 1.0, 1.0, 0.0]          # genome value where each gene removes nothing
INDOOR = {"bonsai", "counter", "kitchen", "room"}


def scene_paths(scene):
    images = "images_2" if scene in INDOOR else "images_4"
    return f"pretrained/nb/gaussian-splatting/mipnerf360/{scene}/point_cloud.ply", f"data/360_v2/{scene}", images


def importance_scores(scene_name, scene, cams, train_idx, mode="sum"):
    """Per-Gaussian importance over all train views at half resolution; cached per scene and mode.
    sum: blend weight summed over views   max: the largest single-view blend weight
    density: the largest per pixel weight (blend weight / footprint area)
    lg: LightGaussian-style hits x opacity x volume (a baseline score)
    learned: gradient learned keep gates (see scores.py)   fisher: photo loss sensitivity (see scores.py)
    """
    suffix = "" if mode == "sum" else f"_{mode}"
    cache = Path("results/cache") / f"{scene_name}_importance{suffix}.pt"
    if cache.exists():
        return torch.load(cache, map_location=scene["means"].device)
    views = cams.subset(train_idx).scaled(0.5)

    if mode == "sum":
        score = gaussian_contributions(scene, views, batch_size=4)["blend_weight"]
    elif mode == "lg":
        hits = gaussian_contributions(scene, views, batch_size=4)["visible_views"]
        volume = scene["scales"].prod(dim=-1)
        volume = (volume / _quantile(volume, 0.9)).clamp(max=1) ** 0.1
        score = hits * scene["opacities"] * volume
    elif mode in ("learned", "fisher"):
        from easplatting.scores import fisher_scores, learned_gates, load_photos

        # a spread of 48 train views keeps these two affordable
        views = views.subset(np.linspace(0, len(views) - 1, min(48, len(views))).astype(int))
        if mode == "learned":
            score = learned_gates(scene, views, render_views(scene, views))
        else:
            score = fisher_scores(scene, views, load_photos(views, scene["means"].device))
    else:
        score = per_view_contributions(scene, views)[
            {"max": "max_weight", "density": "max_density"}[mode]
        ]
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(score.cpu(), cache)
    return score


def setup_scene(scene_name, genes, importance="sum"):
    """Load the scene + cameras and build the gene statistics and bounds (4 genes also computes/loads importance)."""
    scene_path, cameras_path, images = scene_paths(scene_name)
    scene = activate(load_scene(scene_path, device="cuda"))
    cams = load_cameras(cameras_path, images=images)
    train_idx, test_idx = select_indices(len(cams), "train"), select_indices(len(cams), "test")
    scores = importance_scores(scene_name, scene, cams, train_idx, importance) if genes == 4 else None
    stats = gene_stats(scene, scores.cuda() if scores is not None else None)
    b = slider_bounds(stats)
    return dict(scene=scene, cams=cams, stats=stats, train_idx=train_idx, test_idx=test_idx,
                bounds=[b["min_opacity"], b["max_scale"], b["max_anisotropy"]])


class Log:
    """Wraps a fitness function: counts evaluations and keeps one row per call."""

    def __init__(self, fitness_function, method, seed):
        self.fn, self.method, self.seed, self.rows = fitness_function, method, seed, []

    def __call__(self, genome):
        t0 = time.perf_counter()
        fitness, kept, psnr = self.fn(genome)
        self.rows.append(dict(method=self.method, seed=self.seed, eval=len(self.rows), genome=[round(float(g), 5) for g in genome],
                              fitness=float(fitness), kept=float(kept), psnr=float(psnr), ms=1000 * (time.perf_counter() - t0)))
        return fitness, kept, psnr


def run_ga(fit, n, args):
    evolutionary_algorithm(fitness_function=fit, total_generations=args.generations, num_parents=args.parents,
                           num_children=args.children, genome_length=n, mutation_size=args.mutation)


def run_random(fit, n, args):
    for _ in range(args.parents + args.children * args.generations):
        fit(np.random.rand(n))


def run_sweep(fit, n, gene, points, refine=0):
    for g in np.linspace(0, 1, points):
        genome = np.array(OFF[:n])
        genome[gene] = g
        fit(genome)
    refine_sweep(fit, n, gene, points, refine)


def refine_sweep(fit, n, gene, points, steps):
    # bisect between the best feasible sweep point and its neighbour that prunes more
    if steps == 0:
        return
    feasible = [row for row in fit.rows if row["fitness"] >= 0]
    if not feasible:
        return
    low = max(feasible, key=lambda row: row["fitness"])["genome"][gene]
    direction = 1 if GENES[gene] in ("opacity", "importance") else -1
    high = min(max(low + direction / (points - 1), 0), 1)
    for _ in range(steps):
        middle = (low + high) / 2
        genome = np.array(OFF[:n])
        genome[gene] = middle
        if fit(genome)[0] >= 0:
            low = middle
        else:
            high = middle


def run_cmaes(fit, n, args):
    population_size = 32
    cma_es(
        fitness_function=fit,
        total_generations=(args.parents + args.children * args.generations)
        // population_size,
        population_size=population_size,
        genome_length=n,
    )


def tau_from_loss(views, baselines, allowed_loss, margin=2.0):
    # PSNR vs the unpruned render that costs about allowed_loss dB against the photos
    W, H = views.size
    photos = torch.stack([load_image(path, size=(W, H)) for path in views.image_paths])
    model_psnr = psnr_fast(baselines, photos.to(baselines.device)).mean().item()
    return model_psnr - 10 * np.log10(10 ** (allowed_loss / 10) - 1) + margin


def view_psnrs(scene, stats, bounds, genome, views, baselines):
    """Per-view PSNR of a genome's pruned scene vs the unpruned baseline renders of `views`."""
    mask = keep_mask(stats, *decode(genome, bounds))
    return psnr_fast(render_views(prune(scene, mask), views, batch_size=4), baselines, max_db=60)


def best_row(rows):
    return max(rows, key=lambda r: r["fitness"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="bonsai")
    ap.add_argument("--genes", type=int, default=4, choices=[3, 4], help="3 thresholds, or + the importance gene")
    ap.add_argument("--methods", default="ga,random,sweep")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=40.0, help="mean PSNR (dB) vs the unpruned renders")
    ap.add_argument("--views", type=int, default=8, help="number of fixed train views the fitness uses (half resolution)")
    ap.add_argument("--fit-scale", type=float, default=0.5, help="fitness view resolution as a fraction of full (0.5 = half, the default; 1 = full)")
    ap.add_argument("--worst-tau", type=float, default=None, help="also require this PSNR on the worst fitness view")
    ap.add_argument("--parents", type=int, default=32)
    ap.add_argument("--children", type=int, default=64)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--mutation", type=float, default=0.1)
    ap.add_argument("--sweep-points", type=int, default=64)
    ap.add_argument("--sweep-refine", type=int, default=0, help="bisection steps after each 1-D sweep (0 = off)")
    ap.add_argument("--importance", default="sum", choices=["sum", "max", "density", "lg", "learned", "fisher"], help="importance score of the 4th gene")
    ap.add_argument("--view-select", default="even", choices=["even", "coverage"], help="how the fitness views are picked")
    ap.add_argument("--allowed-loss", type=float, default=None, help="derive tau from this allowed real loss (dB) vs the photos")
    ap.add_argument("--out", default=None, help="default results/<scene>_<genes>g")
    ap.add_argument("--publish-images", action="store_true", help="also commit-ready copy of the full-resolution images (~17 MB) in figures/")
    args = ap.parse_args()
    n = args.genes
    tag = (f"_v{args.views}" if args.views != 8 else "") + (f"_w{args.worst_tau:g}" if args.worst_tau else "")
    tag += (f"_i{args.importance}" if args.importance != "sum" else "") + ("_cov" if args.view_select == "coverage" else "")
    tag += f"_L{args.allowed_loss:g}" if args.allowed_loss else ""
    tag += f"_res{args.fit_scale:g}" if args.fit_scale != 0.5 else ""
    out = Path(args.out or f"results/{args.scene}_{n}g{tag}")
    out.mkdir(parents=True, exist_ok=True)

    ctx = setup_scene(args.scene, n, args.importance)
    scene, cams, stats, bounds, train_idx, test_idx = (ctx[k] for k in ("scene", "cams", "stats", "bounds", "train_idx", "test_idx"))
    if args.view_select == "coverage":
        view_idx = coverage_indices(cams, train_idx, args.views)
    else:
        view_idx = select_indices(len(cams), "train", k=args.views)
    views = cams.subset(view_idx).scaled(args.fit_scale)   # fixed fitness views (half resolution unless --fit-scale)
    baselines = render_views(scene, views)
    if args.allowed_loss:
        args.tau = round(tau_from_loss(views, baselines, args.allowed_loss), 2)
        print(f"tau {args.tau} dB = allowed real loss {args.allowed_loss} dB vs the photos", flush=True)
    fitness_function = make_fitness_function(scene, stats, bounds, views, baselines, args.tau, args.worst_tau)
    # held-out diagnostics for each run's best genome: all train views (at the fitness resolution; 48 spread views
    # when that is above half, for memory) and the test views (full res)
    held_idx = train_idx if args.fit_scale <= 0.5 else [train_idx[i] for i in select_indices(len(train_idx), "all", k=48)]
    all_train, test = cams.subset(held_idx).scaled(args.fit_scale), cams.subset(test_idx)
    all_train_base, test_base = render_views(scene, all_train, batch_size=4), render_views(scene, test, batch_size=4)

    runs = []                                         # (method, seed)
    for m in args.methods.split(","):
        if m == "sweep":
            runs += [(f"sweep_{GENES[g]}", 0) for g in range(n)]
        else:
            runs += [(m, s) for s in range(args.seeds)]

    rows, summary = [], []
    for method, seed in runs:
        np.random.seed(seed)
        fit = Log(fitness_function, method, seed)
        t0 = time.perf_counter()
        if method == "ga":
            run_ga(fit, n, args)
        elif method == "random":
            run_random(fit, n, args)
        elif method == "cmaes":
            run_cmaes(fit, n, args)
        else:
            run_sweep(fit, n, GENES.index(method[len("sweep_"):]), args.sweep_points, args.sweep_refine)
        best = best_row(fit.rows)
        feasible = best["fitness"] >= 0
        s = dict(method=method, seed=seed, evals=len(fit.rows), seconds=round(time.perf_counter() - t0, 1), feasible=feasible,
                 genome=best["genome"], kept=best["kept"], fit_psnr=best["psnr"])
        if feasible:
            tr, te = (view_psnrs(scene, stats, bounds, best["genome"], v, b) for v, b in ((all_train, all_train_base), (test, test_base)))
            s.update(train_psnr=tr.mean().item(), train_worst=tr.min().item(), test_psnr=te.mean().item(), test_worst=te.min().item())
            s["thresholds"] = [float(f"{t:.4g}") for t in decode(best["genome"], bounds)]
        summary.append(s)
        rows += fit.rows
        print(f"{method:>18} seed {seed}: {s['evals']:4d} evals  kept {100 * s['kept']:5.1f}%  fit PSNR {s['fit_psnr']:5.2f}"
              + (f"  all-train {s['train_psnr']:5.2f} (worst {s['train_worst']:5.2f})  test {s['test_psnr']:5.2f} (worst {s['test_worst']:5.2f})"
                 if feasible else "  INFEASIBLE"), flush=True)

    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(dict(args=vars(args), runs=summary), indent=1))
    print(f"\n{args.scene}, {n} genes, tau {args.tau} dB: mean best kept % over seeds (lower is better)")
    for method in dict.fromkeys(m for m, _ in runs):
        k = [100 * s["kept"] for s in summary if s["method"] == method and s["feasible"]]
        print(f"{method:>18}  {np.mean(k):5.1f}  (best {min(k):.1f}, {len(k)} feasible runs)" if k else f"{method:>18}  no feasible run")
    from easplatting.plot_run import publish
    from easplatting.strip import render_strip
    render_strip(out, ctx)
    print(f"wrote {out}/log.jsonl and summary.json; figures: {', '.join(str(p) for p in publish(out, images=args.publish_images))}")


if __name__ == "__main__":
    main()
