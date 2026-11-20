"""rescore.py — re-score the best genomes of finished runs at full resolution (no new search).

    python -m easplatting.rescore --runs counter_4g_v32_w35_imax bonsai_4g_v32_w35_imax

For every feasible run in results/<run>/summary.json, render the pruned scene and the unpruned scene on
  fit-full    the 32 fitness views at FULL resolution (the fitness views the search used at half resolution)
  test        the held-out test views (full resolution)
and report PSNR vs the unpruned render: mean, worst view, and whether the fitness floors (the run's tau and worst-tau) still hold.
Output: results/<run>/rescore.json (one row per run) and a table on stdout (mean over seeds per method).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from easplatting.experiment import setup_scene, view_psnrs
from easplatting.render import render_views, select_indices

TOL = 0.02  # dB of slack on a floor, the same as a rounding error


def rescore_run(run, ctx, cache):
    path = Path("results") / run / "summary.json"
    summary = json.loads(path.read_text())
    args = summary["args"]
    scene, cams, stats, bounds = (ctx[k] for k in ("scene", "cams", "stats", "bounds"))
    tau, worst_tau = args["tau"], args.get("worst_tau")

    # unpruned renders of the two view sets, shared by every run on this scene
    if "sets" not in cache:
        fit_idx = select_indices(len(cams), "train", k=args["views"])
        sets = {"fit-full": cams.subset(fit_idx), "test": cams.subset(ctx["test_idx"])}
        cache["sets"] = {name: (v, render_views(scene, v, batch_size=4)) for name, v in sets.items()}

    rows = []
    for s in summary["runs"]:
        if not s["feasible"]:
            continue
        row = dict(run=run, method=s["method"], seed=s["seed"], kept=s["kept"], fit_psnr=s["fit_psnr"])
        for name, (views, base) in cache["sets"].items():
            p = view_psnrs(scene, stats, bounds, s["genome"], views, base)
            row[f"{name}_mean"], row[f"{name}_worst"] = p.mean().item(), p.min().item()
        row["mean_ok"] = row["fit-full_mean"] >= tau - TOL
        row["worst_ok"] = worst_tau is None or row["fit-full_worst"] >= worst_tau - TOL
        row["floors_ok"] = row["mean_ok"] and row["worst_ok"]
        rows.append(row)
    (Path("results") / run / "rescore.json").write_text(json.dumps(rows, indent=1))
    return rows, args


def print_table(rows, args):
    print(f"\n{args['scene']}  floors: mean >= {args['tau']} dB" + (f", worst >= {args['worst_tau']} dB" if args.get("worst_tau") else ""))
    print(f"{'method':>16} {'runs':>4} {'kept%':>6} {'run fit':>8} {'full fit mean/worst':>20} {'test mean/worst':>16} {'floors held':>11}")
    for method in dict.fromkeys(r["method"] for r in rows):
        r = [x for x in rows if x["method"] == method]
        m = lambda key: np.mean([x[key] for x in r])
        print(f"{method:>16} {len(r):>4} {100 * m('kept'):>6.1f} {m('fit_psnr'):>8.2f} "
              f"{m('fit-full_mean'):>9.2f} /{m('fit-full_worst'):>7.2f} {m('test_mean'):>8.2f} /{m('test_worst'):>6.2f} "
              f"{sum(x['floors_ok'] for x in r):>6}/{len(r)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folder names under results/")
    a = ap.parse_args()
    caches = {}  # one scene context and unpruned renders at a time (the GPU holds only one scene)
    for run in a.runs:
        args = json.loads((Path("results") / run / "summary.json").read_text())["args"]
        if args["scene"] not in caches:
            caches.clear()
            torch.cuda.empty_cache()
            caches[args["scene"]] = dict(ctx=setup_scene(args["scene"], args["genes"], args["importance"]))
        cache = caches[args["scene"]]
        rows, args = rescore_run(run, cache["ctx"], cache)
        print_table(rows, args)


if __name__ == "__main__":
    main()
