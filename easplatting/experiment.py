"""experiment.py — GA vs baselines on one scene, one JSONL row per fitness evaluation.

    python -m easplatting.experiment --scene bonsai --genes 4 --seeds 3

Methods (all minimise kept fraction subject to PSNR >= tau vs the unpruned render, same fitness function):
  ga           evolutionary_algorithm (ea.py), parents + children x generations evaluations
  random       the same number of uniform random genomes
  sweep_<gene> 1-D sweep of one gene with every other gene off (deterministic, run once)
Output: results/<scene>_<genes>g/log.jsonl (every evaluation) and summary.json (best per run, held-out test PSNR).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from easplatting.cameras import load_cameras
from easplatting.ea import evolutionary_algorithm
from easplatting.fitness import make_fitness_function
from easplatting.genome import decode
from easplatting.gs_loader import activate, load_scene, prune
from easplatting.metrics import psnr_fast
from easplatting.render import gaussian_contributions, render_views, select_indices
from easplatting.thresholds import gene_stats, keep_mask, slider_bounds

GENES = ["opacity", "scale", "anisotropy", "importance"]
OFF = [0.0, 1.0, 1.0, 0.0]          # genome value where each gene removes nothing
INDOOR = {"bonsai", "counter", "kitchen", "room"}


def scene_paths(scene):
    images = "images_2" if scene in INDOOR else "images_4"
    return f"pretrained/nb/gaussian-splatting/mipnerf360/{scene}/point_cloud.ply", f"data/360_v2/{scene}", images


def importance_scores(scene_name, scene, cams, train_idx):
    """Per-Gaussian blend weight over all train views at half resolution; cached per scene."""
    cache = Path("results/cache") / f"{scene_name}_importance.pt"
    if cache.exists():
        return torch.load(cache, map_location=scene["means"].device)
    w = gaussian_contributions(scene, cams.subset(train_idx).scaled(0.5), batch_size=4)["blend_weight"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w.cpu(), cache)
    return w


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


def run_sweep(fit, n, gene, points):
    for g in np.linspace(0, 1, points):
        genome = np.array(OFF[:n])
        genome[gene] = g
        fit(genome)


def test_psnr(scene, stats, bounds, genome, cams, test_idx):
    """Held-out check: best genome vs the unpruned render on the test views, full resolution."""
    test = cams.subset(test_idx)
    mask = keep_mask(stats, *decode(genome, bounds))
    return psnr_fast(render_views(prune(scene, mask), test, batch_size=4),
                     render_views(scene, test, batch_size=4), max_db=60).mean().item()


def best_row(rows):
    return max(rows, key=lambda r: r["fitness"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="bonsai")
    ap.add_argument("--genes", type=int, default=4, choices=[3, 4], help="3 thresholds, or + the importance gene")
    ap.add_argument("--methods", default="ga,random,sweep")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=40.0, help="PSNR (dB) vs the unpruned renders")
    ap.add_argument("--parents", type=int, default=32)
    ap.add_argument("--children", type=int, default=64)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--mutation", type=float, default=0.1)
    ap.add_argument("--sweep-points", type=int, default=64)
    ap.add_argument("--out", default=None, help="default results/<scene>_<genes>g")
    args = ap.parse_args()
    n = args.genes
    out = Path(args.out or f"results/{args.scene}_{n}g")
    out.mkdir(parents=True, exist_ok=True)

    scene_path, cameras_path, images = scene_paths(args.scene)
    scene = activate(load_scene(scene_path, device="cuda"))
    cams = load_cameras(cameras_path, images=images)
    train_idx, test_idx = select_indices(len(cams), "train"), select_indices(len(cams), "test")
    importance = importance_scores(args.scene, scene, cams, train_idx) if n == 4 else None
    stats = gene_stats(scene, importance.cuda() if importance is not None else None)
    b = slider_bounds(stats)
    bounds = [b["min_opacity"], b["max_scale"], b["max_anisotropy"]]
    views = cams.subset(select_indices(len(cams), "train", k=8)).scaled(0.5)   # fixed fitness views, half resolution
    fitness_function = make_fitness_function(scene, stats, bounds, views, render_views(scene, views), args.tau)

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
        else:
            run_sweep(fit, n, GENES.index(method[len("sweep_"):]), args.sweep_points)
        best = best_row(fit.rows)
        feasible = best["fitness"] >= 0
        s = dict(method=method, seed=seed, evals=len(fit.rows), seconds=round(time.perf_counter() - t0, 1), feasible=feasible,
                 genome=best["genome"], kept=best["kept"], fit_psnr=best["psnr"])
        if feasible:
            s["test_psnr"] = test_psnr(scene, stats, bounds, best["genome"], cams, test_idx)
            s["thresholds"] = [float(f"{t:.4g}") for t in decode(best["genome"], bounds)]
        summary.append(s)
        rows += fit.rows
        print(f"{method:>18} seed {seed}: {s['evals']:4d} evals  kept {100 * s['kept']:5.1f}%  fit PSNR {s['fit_psnr']:5.2f}"
              + (f"  test PSNR {s['test_psnr']:5.2f}" if feasible else "  INFEASIBLE"), flush=True)

    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(dict(args=vars(args), runs=summary), indent=1))
    print(f"\n{args.scene}, {n} genes, tau {args.tau} dB: mean best kept % over seeds (lower is better)")
    for method in dict.fromkeys(m for m, _ in runs):
        k = [100 * s["kept"] for s in summary if s["method"] == method and s["feasible"]]
        print(f"{method:>18}  {np.mean(k):5.1f}  (best {min(k):.1f}, {len(k)} feasible runs)" if k else f"{method:>18}  no feasible run")
    print(f"wrote {out}/log.jsonl and summary.json")


if __name__ == "__main__":
    main()
