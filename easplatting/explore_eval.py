"""explore_eval.py — held-out evaluation of the genomes found by explore.py (no search; each cell is a few renders).

    python -m easplatting.explore_eval transfer --targets bonsai,counter,garden,bicycle
    python -m easplatting.explore_eval fronts   --scene bonsai
    python -m easplatting.explore_eval stress   --scene bonsai
transfer: apply a genome evolved on scene X to scene Y (features are percentile ranks and shells are distance quantiles, so a genome is
          scene independent). Compared with the 1-D sweep and a fixed importance cutoff of 0.5 on Y.
fronts:   size vs quality on the TEST views against the photos (PSNR / SSIM / LPIPS-vgg, what papers report) for several scores
          (max, lg = LightGaussian-style, density, sum, opacity, ...) and for the evolved genomes.
stress:   PSNR vs the unpruned render from cameras the train views never saw (jittered poses), to see whether importance scores computed
          on train views leave holes elsewhere.
Reads results/explore/results.jsonl, writes results/explore/{transfer,fronts,stress}.jsonl.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from easplatting.cameras import Cameras, load_image
from easplatting.explore import (
    feature_rank,
    make_mask_fn,
    make_fitness,
    raw_feature,
    region_ids,
    sweep_cutoff,
)
from easplatting.experiment import Log, setup_scene
from easplatting.gs_loader import prune
from easplatting.metrics import ImageMetrics, psnr_fast, summarize
from easplatting.render import render_views, select_indices

OUT = Path("results/explore")
TAU, WORST = 40.0, 35.0
TOL = 0.02  # dB: genomes on the boundary recompute to 39.9999


class Target:
    """One scene with the fitness views, the test views and their unpruned renders."""

    def __init__(self, name, views=32):
        self.name = name
        self.ctx = setup_scene(name, 4, "max")
        self.scene, cams = self.ctx["scene"], self.ctx["cams"]
        self.fit = cams.subset(select_indices(len(cams), "train", k=views)).scaled(0.5)
        self.fit_base = render_views(self.scene, self.fit)
        self.test = cams.subset(self.ctx["test_idx"])
        self.test_base = render_views(self.scene, self.test, batch_size=4)
        self.ranks, self.regions, self.photos, self.metrics = {}, {}, None, None

    def feature(self, name):
        if name not in self.ranks:
            self.ranks[name] = feature_rank(name, raw_feature(name, self.ctx, self.name).float())
        return self.ranks[name]

    def mask_from_row(self, row, scale=1.0):
        """scale < 1 or > 1 multiplies every cutoff (calibration: same shape, different strength)."""
        names = row["features"].split(",")
        feats = torch.stack([self.feature(n) for n in names], dim=1)
        if row["regions"] not in self.regions:
            self.regions[row["regions"]] = region_ids(row["regions"], self.ctx, self.name)
        regions, count = self.regions[row["regions"]]
        fn, _ = make_mask_fn(feats, regions, count, row["learn_weights"], names.index("max"), row["local"])
        genome = np.array(row["genome"], dtype=float)
        k = len(names) if row["learn_weights"] else 0
        genome[k:] = np.clip(genome[k:] * scale, 0, 1)
        return fn(genome)

    def feasible(self, mask):
        pruned = prune(self.scene, mask)
        f = psnr_fast(render_views(pruned, self.fit), self.fit_base, max_db=60)
        return f.mean().item() >= TAU - TOL and f.min().item() >= WORST - TOL

    def calibrate(self, row, steps=14):
        """Largest cutoff scale whose mask still meets both floors on this scene's fitness views (bisection)."""
        low, high = 0.0, 1.6
        if not self.feasible(self.mask_from_row(row, 1.0)):
            high = 1.0
        else:
            low = 1.0
        for _ in range(steps):
            mid = (low + high) / 2
            if self.feasible(self.mask_from_row(row, mid)):
                low = mid
            else:
                high = mid
        return low
    def score_mask(self, name, kept):
        """Keep the top `kept` fraction by a score ('opacity' and the explore features; higher = more important)."""
        r = self.feature(name)
        return r >= 1 - kept

    def evaluate(self, mask, photos=False):
        out = dict(kept=mask.float().mean().item())
        pruned = prune(self.scene, mask)
        f = psnr_fast(render_views(pruned, self.fit), self.fit_base, max_db=60)
        rendered = render_views(pruned, self.test, batch_size=4)
        t = psnr_fast(rendered, self.test_base, max_db=60)
        out.update(fit_mean=f.mean().item(), fit_worst=f.min().item(), test_mean=t.mean().item(), test_worst=t.min().item())
        out["fit_ok"] = out["fit_mean"] >= TAU - TOL and out["fit_worst"] >= WORST - TOL
        out["test_ok"] = out["test_mean"] >= TAU - TOL and out["test_worst"] >= WORST - TOL
        if photos:
            out.update(self.vs_photos(rendered))
        return out

    def vs_photos(self, rendered):
        if self.photos is None:
            W, H = self.test.size
            self.photos = torch.stack([load_image(p, size=(W, H)) for p in self.test.image_paths]).cuda()
            self.metrics = ImageMetrics(device="cuda", lpips_net="vgg")
        s = summarize(self.metrics(rendered, self.photos))
        return dict(photo_psnr=s["psnr"], photo_ssim=s["ssim"], photo_lpips=s["lpips"])


def references(target):
    """The two baselines on this scene: the refined 1-D importance sweep, and a fixed cutoff 0.5 on max."""
    feats = torch.stack([target.feature("max")], dim=1)
    zeros = torch.zeros(feats.shape[0], dtype=torch.long, device="cuda")
    fn, _ = make_mask_fn(feats, zeros, 1, False, 0)
    fit = Log(make_fitness(target.scene, fn, target.fit, target.fit_base, TAU, WORST), "sweep", 0)
    cutoff = sweep_cutoff(fit)
    return {"sweep": target.evaluate(fn(np.array([cutoff]))), "fixed0.5": target.evaluate(fn(np.array([0.5])))}, cutoff


def load_rows(methods=("ga", "cmaes")):
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines()]
    return [r for r in rows if r["method"] in methods and r["feasible"] and r["regions"] != "cell8"]


def have_features(row, scene):
    return all((Path("results/cache") / f"{scene}_importance_{n}.pt").exists() for n in row["features"].split(",") if n in ("fisherj", "learned2"))


def cmd_transfer(a):
    rows = load_rows()
    out = []
    for scene in a.targets.split(","):
        target = Target(scene)
        refs, cutoff = references(target)
        for k, v in refs.items():
            out.append(dict(target=scene, source="-", name=k, method=k, seed=0, **v))
        print(f"{scene}: sweep kept {100 * refs['sweep']['kept']:.1f}% (test {refs['sweep']['test_mean']:.1f}/{refs['sweep']['test_worst']:.1f}, "
              f"fit_ok {refs['sweep']['fit_ok']}); fixed 0.5 kept 50% test {refs['fixed0.5']['test_mean']:.1f}/{refs['fixed0.5']['test_worst']:.1f}", flush=True)
        for r in rows:
            if not have_features(r, scene):
                continue
            scale = target.calibrate(r) if a.calibrate else 1.0
            ev = target.evaluate(target.mask_from_row(r, scale))
            out.append(dict(target=scene, source=r["scene"], name=r["name"], method=r["method"], seed=r["seed"], scale=scale, **ev))
            del ev
        torch.cuda.empty_cache()
        del target
    with open(OUT / ("transfer_calibrated.jsonl" if a.calibrate else "transfer.jsonl"), "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in out)
    summarize_transfer(out)


def summarize_transfer(out):
    print(f"\n{'target':<8}{'source':<8}{'config':<16}{'method':<7}{'n':>2} {'kept%':>6} {'fit_ok':>7} {'test_ok':>8} {'test dB':>8} {'worst':>6}")
    keys = dict.fromkeys((o["target"], o["source"], o["name"], o["method"]) for o in out)
    for key in keys:
        g = [o for o in out if (o["target"], o["source"], o["name"], o["method"]) == key]
        m = lambda k: np.mean([o[k] for o in g])
        print(f"{key[0]:<8}{key[1]:<8}{key[2]:<16}{key[3]:<7}{len(g):>2} {100 * m('kept'):6.1f} {m('fit_ok'):7.2f} {m('test_ok'):8.2f} {m('test_mean'):8.2f} {m('test_worst'):6.2f}")


def cmd_fronts(a):
    target = Target(a.scene)
    out = []
    unpruned = target.vs_photos(target.test_base)
    out.append(dict(scene=a.scene, kind="unpruned", label="unpruned", kept=1.0, **unpruned))
    print(f"unpruned vs photos: {unpruned}", flush=True)
    scores = a.scores.split(",")
    for s in scores:
        for kept in np.arange(0.9, 0.25, -0.05):
            ev = target.evaluate(target.score_mask(s, float(kept)), photos=True)
            out.append(dict(scene=a.scene, kind="score", label=s, **ev))
            print(f"{s:>9} kept {100 * kept:4.0f}%  test {ev['test_mean']:5.1f}/{ev['test_worst']:5.1f}  vs photos {ev['photo_psnr']:.2f} "
                  f"ssim {ev['photo_ssim']:.4f} lpips {ev['photo_lpips']:.4f}", flush=True)
    for r in load_rows():
        if r["scene"] != a.scene or not have_features(r, a.scene):
            continue
        ev = target.evaluate(target.mask_from_row(r), photos=True)
        out.append(dict(scene=a.scene, kind="evolved", label=f"{r['name']}/{r['method']}", **ev))
        print(f"{r['name']:>14}/{r['method']:<6} kept {100 * ev['kept']:5.1f}%  test {ev['test_mean']:5.1f}/{ev['test_worst']:5.1f}  vs photos "
              f"{ev['photo_psnr']:.2f} ssim {ev['photo_ssim']:.4f} lpips {ev['photo_lpips']:.4f}", flush=True)
    with open(OUT / "fronts.jsonl", "a") as f:
        f.writelines(json.dumps(o) + "\n" for o in out)


def jitter_cameras(cams, translate, degrees, seed=0):
    """Same cameras, perturbed: the centre moves by a random vector of length `translate`, the view turns by `degrees` about a random axis."""
    g = torch.Generator().manual_seed(seed)
    vm = cams.viewmats.clone().cpu()
    for i in range(len(vm)):
        R, t = vm[i, :3, :3], vm[i, :3, 3]
        centre = -R.T @ t
        d = torch.randn(3, generator=g)
        centre = centre + translate * d / d.norm()
        axis = torch.randn(3, generator=g)
        axis = axis / axis.norm()
        angle = torch.tensor(np.deg2rad(degrees)) * torch.rand(1, generator=g)
        K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        turn = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        R2 = turn @ R
        vm[i, :3, :3], vm[i, :3, 3] = R2, -R2 @ centre
    return Cameras(cams.names, vm, cams.Ks, cams.widths, cams.heights, cams.image_paths)


def cmd_stress(a):
    target = Target(a.scene)
    train = target.ctx["cams"].subset(target.ctx["train_idx"])
    centres = train.centers()
    spacing = torch.cdist(centres, centres).fill_diagonal_(float("inf")).amin(dim=1).median().item()
    print(f"{a.scene}: median spacing between train cameras {spacing:.3f}", flush=True)
    refs, cutoff = references(target)
    base_mask = target.feature("max") >= cutoff
    kept = base_mask.float().mean().item()
    masks = {f"max cutoff (kept {100 * kept:.1f}%)": base_mask, "lg, same kept": target.score_mask("lg", kept),
             "opacity, same kept": target.score_mask("opacity", kept)}
    for r in load_rows(("cmaes",)):
        if r["scene"] == a.scene and r["seed"] == 0 and have_features(r, a.scene):
            masks[f"{r['name']} (kept {100 * r['kept']:.1f}%)"] = target.mask_from_row(r)
    levels = [("test views", 0.0, 0.0), ("jitter 0.25 spacing, 2 deg", 0.25, 2.0), ("jitter 0.5 spacing, 5 deg", 0.5, 5.0), ("jitter 1.0 spacing, 10 deg", 1.0, 10.0)]
    out = []
    for level, trans, deg in levels:
        cams = target.test if trans == 0 else jitter_cameras(train.subset(select_indices(len(train), "all", k=24)), trans * spacing, deg, seed=1)
        base = target.test_base if trans == 0 else render_views(target.scene, cams, batch_size=4)
        for label, mask in masks.items():
            p = psnr_fast(render_views(prune(target.scene, mask), cams, batch_size=4), base, max_db=60)
            out.append(dict(scene=a.scene, level=level, label=label, mean=p.mean().item(), worst=p.min().item(), p10=p.quantile(0.1).item()))
            print(f"{level:<28}{label:<34} mean {p.mean():6.2f}  p10 {p.quantile(0.1):6.2f}  worst {p.min():6.2f}", flush=True)
    with open(OUT / "stress.jsonl", "a") as f:
        f.writelines(json.dumps(o) + "\n" for o in out)


def cmd_gapcheck(a):
    """PSNR vs the unpruned render on {train, test} x {half, full} resolution: is the test gap held-out poses or resolution?"""
    target = Target(a.scene)
    cams = target.ctx["cams"]
    sets = {f"{split} {res}": cams.subset(target.ctx[f"{split}_idx"]).scaled(0.5 if res == "half" else 1.0)
            for split in ("train", "test") for res in ("half", "full")}
    sets["train full"] = sets["train full"].subset(select_indices(len(sets["train full"]), "all", k=48))  # 48 spread views: memory
    bases = {k: render_views(target.scene, v, batch_size=4) for k, v in sets.items()}
    cases = {"sweep (max cutoff)": target.feature("max") >= references(target)[1]}
    for r in load_rows(("cmaes",)):
        if r["scene"] == a.scene and r["name"] in a.names.split(",") and r["seed"] == 0 and have_features(r, a.scene):
            cases[r["name"]] = target.mask_from_row(r)
    print(f"{a.scene}: mean / worst PSNR vs unpruned (dB)", flush=True)
    print(f"{'':<22}" + "".join(f"{k:>18}" for k in sets), flush=True)
    for label, mask in cases.items():
        pruned = prune(target.scene, mask)
        cells = []
        for k, v in sets.items():
            p = psnr_fast(render_views(pruned, v, batch_size=4), bases[k], max_db=60)
            cells.append(f"{p.mean():9.2f} /{p.min():6.1f}")
        print(f"{label:<22}" + "".join(f"{c:>18}" for c in cells), flush=True)


def cmd_fps(a):
    """FPS of the actual evolved masks (seed 0, CMA-ES) against the unpruned scene, the 1-D sweep optimum and opacity pruning at the same size."""
    from easplatting.render import measure_fps

    target = Target(a.scene)
    cutoff = references(target)[1]
    cases = {"unpruned": torch.ones(len(target.feature("max")), dtype=torch.bool, device="cuda"),
             "sweep (max cutoff)": target.feature("max") >= cutoff}
    for r in load_rows(("cmaes",)):
        if r["scene"] == a.scene and r["name"] in a.names.split(",") and r["seed"] == 0 and have_features(r, a.scene):
            cases[f"evolved {r['name']}"] = target.mask_from_row(r)
            cases[f"opacity, kept {100 * r['kept']:.1f}%"] = target.score_mask("opacity", r["kept"])
    base = None
    for label, mask in cases.items():
        fps = measure_fps(prune(target.scene, mask), target.test, n_warmup=20, n_iters=100)
        base = base or fps["fps"]
        print(f"{a.scene:>8} {label:<28} kept {100 * mask.float().mean().item():5.1f}%  {fps['ms_per_frame']:6.2f} ms  {fps['fps']:7.1f} fps  x{fps['fps'] / base:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transfer")
    t.add_argument("--targets", default="bonsai,counter,garden,bicycle")
    t.add_argument("--calibrate", action="store_true", help="rescale all cutoffs per target so the floors are met on its fitness views")
    f = sub.add_parser("fronts")
    f.add_argument("--scene", default="bonsai")
    f.add_argument("--scores", default="max,lg,density,sum,opacity")
    p = sub.add_parser("fps")
    p.add_argument("--scene", default="bonsai")
    p.add_argument("--names", default="wall_warm,n_foot,shell4_warm")
    g = sub.add_parser("gapcheck")
    g.add_argument("--scene", default="bonsai")
    g.add_argument("--names", default="v32_wall,shell4_warm,w5_warm")
    s = sub.add_parser("stress")
    s.add_argument("--scene", default="bonsai")
    a = ap.parse_args()
    {"transfer": cmd_transfer, "fronts": cmd_fronts, "stress": cmd_stress, "gapcheck": cmd_gapcheck, "fps": cmd_fps}[a.cmd](a)


if __name__ == "__main__":
    main()
