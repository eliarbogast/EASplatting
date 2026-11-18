"""front.py — size/quality Pareto fronts: NSGA-II and MAP-Elites vs random and sweeps.

    python -m easplatting.front --scene bonsai --views 32 --importance max

Every method logs every evaluation as (kept, mean PSNR, worst-view PSNR, importance
share). Quality is the worst fitness-view PSNR (--quality mean for the mean), clipped to
[20, 60] dB. A front is the non-dominated set of (kept, quality) over a run's
evaluations. Hypervolume is the area it dominates up to the reference point (kept = 1,
quality = 20 dB) divided by the whole box, so 1.0 would mean nothing kept at 60 dB.
MAP-Elites keeps the best quality per cell of (kept, importance share).
Output: results/<scene>_front_v<views>[_i<mode>]/ with log.jsonl and summary.json, then
plot_front.publish_front() copies front.png and summary.json into figures/runs/.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from easplatting.experiment import GENES, run_sweep, setup_scene, view_psnrs
from easplatting.fitness import make_objective_function
from easplatting.map_elites import map_elites
from easplatting.nsga2 import front_2d, hypervolume, nsga2
from easplatting.render import coverage_indices, render_views, select_indices

QUALITY_FLOOR = 20.0
QUALITY_CEILING = 60.0
REFERENCE = (1.0, -QUALITY_FLOOR)
BOX = QUALITY_CEILING - QUALITY_FLOOR
BINS = (10, 5)


class ObjectiveLog:
    # wraps an objective function: one row per evaluation
    def __init__(self, objective_function, method, seed):
        self.objective_function = objective_function
        self.method = method
        self.seed = seed
        self.rows = []

    def __call__(self, genome):
        kept, mean, worst, share = self.objective_function(genome)
        self.rows.append(
            dict(
                method=self.method,
                seed=self.seed,
                eval=len(self.rows),
                genome=[round(float(g), 5) for g in genome],
                kept=kept,
                mean=mean,
                worst=worst,
                share=share,
            )
        )
        return kept, mean, worst, share


def quality_of(row, quality):
    return float(np.clip(row[quality], QUALITY_FLOOR, QUALITY_CEILING))


def objectives_of(rows, quality):
    # both minimized: kept fraction and minus the quality
    return np.array([[row["kept"], -quality_of(row, quality)] for row in rows])


def run_nsga2(log, n, args):
    def objectives(genome):
        kept, mean, worst, share = log(genome)
        row = dict(mean=mean, worst=worst)
        return np.array([kept, -quality_of(row, args.quality)])

    nsga2(
        objective_function=objectives,
        total_generations=args.generations,
        population_size=args.population,
        genome_length=n,
        mutation_size=args.mutation,
        reference=REFERENCE,
    )


def run_map_elites(log, n, args):
    def fitness_function(genome):
        kept, mean, worst, share = log(genome)
        row = dict(mean=mean, worst=worst)
        quality = quality_of(row, args.quality)
        return quality - QUALITY_FLOOR, np.array([kept, share])

    return map_elites(
        fitness_function=fitness_function,
        total_evaluations=args.population * (args.generations + 1),
        num_initial=args.population,
        genome_length=n,
        mutation_size=args.mutation,
        bins=BINS,
    )


def run_random(log, n, args):
    for _ in range(args.population * (args.generations + 1)):
        log(np.random.rand(n))


def run_sweeps(log, n, args):
    for gene in range(n):
        run_sweep(log, n, gene, args.sweep_points)


def summarize_run(rows, quality, check_every=16):
    objectives = objectives_of(rows, quality)
    front = front_2d(objectives)
    curve = [
        hypervolume(objectives[:k], REFERENCE) / BOX
        for k in range(check_every, len(rows) + 1, check_every)
    ]
    points = [
        dict(
            kept=rows[i]["kept"],
            quality=quality_of(rows[i], quality),
            mean=rows[i]["mean"],
            worst=rows[i]["worst"],
            share=rows[i]["share"],
            genome=rows[i]["genome"],
        )
        for i in front
    ]
    # best quality on the front at a given size
    at = lambda limit: max(
        [p["quality"] for p in points if p["kept"] <= limit], default=QUALITY_FLOOR
    )
    return dict(
        evals=len(rows),
        hypervolume=curve[-1],
        hypervolume_curve=curve,
        check_every=check_every,
        front=points,
        quality_at_50=at(0.5),
        quality_at_60=at(0.6),
    )


def held_out(ctx, test, test_base, points, limit=12):
    # re-measure front points on the test views at full resolution
    if len(points) > limit:
        points = [points[i] for i in np.linspace(0, len(points) - 1, limit).astype(int)]
    results = []
    for point in points:
        views = view_psnrs(
            ctx["scene"], ctx["stats"], ctx["bounds"], point["genome"], test, test_base
        )
        results.append(
            dict(
                kept=point["kept"],
                fit_mean=point["mean"],
                fit_worst=point["worst"],
                test_mean=views.mean().item(),
                test_worst=views.min().item(),
                genome=point["genome"],
            )
        )
    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", default="bonsai")
    ap.add_argument("--methods", default="nsga2,map_elites,random,sweep")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--views", type=int, default=32, help="fitness views (half res)")
    ap.add_argument("--view-select", default="even", choices=["even", "coverage"])
    choices = ["sum", "max", "density", "lg", "learned", "fisher"]
    ap.add_argument("--importance", default="sum", choices=choices)
    ap.add_argument("--quality", default="worst", choices=["worst", "mean"])
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--mutation", type=float, default=0.1)
    ap.add_argument("--sweep-points", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n = 4
    tag = f"_v{args.views}"
    tag += f"_i{args.importance}" if args.importance != "sum" else ""
    tag += "_cov" if args.view_select == "coverage" else ""
    tag += "_mean" if args.quality == "mean" else ""
    out = Path(args.out or f"results/{args.scene}_front{tag}")
    out.mkdir(parents=True, exist_ok=True)

    ctx = setup_scene(args.scene, n, args.importance)
    scene, cams = ctx["scene"], ctx["cams"]
    if args.view_select == "coverage":
        view_idx = coverage_indices(cams, ctx["train_idx"], args.views)
    else:
        view_idx = select_indices(len(cams), "train", k=args.views)
    views = cams.subset(view_idx).scaled(0.5)
    objective_function = make_objective_function(
        scene, ctx["stats"], ctx["bounds"], views, render_views(scene, views)
    )
    test = cams.subset(ctx["test_idx"])
    test_base = render_views(scene, test, batch_size=4)

    runs = []
    for method in args.methods.split(","):
        seeds = 1 if method == "sweep" else args.seeds
        runs += [(method, seed) for seed in range(seeds)]

    rows, summary = [], []
    for method, seed in runs:
        np.random.seed(seed)
        log = ObjectiveLog(objective_function, method, seed)
        start = time.perf_counter()
        extra = {}
        if method == "nsga2":
            run_nsga2(log, n, args)
        elif method == "map_elites":
            coverage, qd_score, archive, _ = run_map_elites(log, n, args)
            archive = np.where(np.isfinite(archive), archive + QUALITY_FLOOR, np.nan)
            extra = dict(
                coverage=coverage[-1],
                qd_score=qd_score[-1],
                archive=archive.tolist(),
                bins=list(BINS),
            )
        elif method == "random":
            run_random(log, n, args)
        else:
            run_sweeps(log, n, args)
        result = summarize_run(log.rows, args.quality)
        seconds = round(time.perf_counter() - start, 1)
        result.update(method=method, seed=seed, seconds=seconds)
        result.update(extra)
        summary.append(result)
        rows += log.rows
        print(
            f"{method:>10} seed {seed}: {result['evals']:4d} evals  "
            f"hypervolume {result['hypervolume']:.4f}  "
            f"front {len(result['front']):3d} pts  "
            f"quality at <=50% kept {result['quality_at_50']:5.1f} dB  "
            f"at <=60% {result['quality_at_60']:5.1f} dB",
            flush=True,
        )

    # held-out check of the best-hypervolume front of every method
    held = {}
    for method in dict.fromkeys(m for m, _ in runs):
        mine = [r for r in summary if r["method"] == method]
        best = max(mine, key=lambda r: r["hypervolume"])
        held[method] = held_out(ctx, test, test_base, best["front"])

    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary_json = dict(args=vars(args), runs=summary, held_out=held)
    (out / "summary.json").write_text(json.dumps(summary_json, indent=1))
    print(f"\n{args.scene}: mean hypervolume over seeds (higher is better)")
    for method in dict.fromkeys(m for m, _ in runs):
        hv = [r["hypervolume"] for r in summary if r["method"] == method]
        print(f"{method:>10}  {np.mean(hv):.4f}  (best {max(hv):.4f}, {len(hv)} runs)")
    from easplatting.plot_front import publish_front

    print(f"wrote {out}; figure: {publish_front(out)}")


if __name__ == "__main__":
    main()
