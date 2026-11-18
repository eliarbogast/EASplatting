"""plot_front.py — the standard figure of a front.py run (CPU only).

    python -m easplatting.plot_front results/<scene>_front_v32

front.png (reads summary.json written by front.py):
  A  size-quality fronts on the fitness views, every seed (higher and further left wins)
  B  hypervolume of the front found so far vs evaluations
  C  the best front of each method re-measured on the held-out test views
  D  the MAP-Elites archive: best quality per cell of (kept, importance share)
Colors: NSGA-II blue, MAP-Elites aqua, random orange, sweeps gray.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from easplatting.plot_run import AQUA, BLUE, GRAY, INK, INK2, ORANGE, SURFACE, _style

COLOR = {"nsga2": BLUE, "map_elites": AQUA, "random": ORANGE, "sweep": GRAY}
LABEL = {
    "nsga2": "NSGA-II",
    "map_elites": "MAP-Elites",
    "random": "random",
    "sweep": "1-D sweeps",
}
FIG_DIR = Path("figures")


def best_run(runs, method):
    mine = [r for r in runs if r["method"] == method]
    return max(mine, key=lambda r: r["hypervolume"])


def plot_front(run_dir, out=None):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    args, runs, held = summary["args"], summary["runs"], summary["held_out"]
    out = Path(out or run_dir / "front.png")
    methods = list(dict.fromkeys(r["method"] for r in runs))

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8.6), facecolor=SURFACE, constrained_layout=True
    )
    (a, b), (c, d) = axes
    title = (
        f"{args['scene']}: size-quality fronts, {args['views']} views, "
        f"{args['quality']}-view quality, {args.get('importance', 'sum')} importance"
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")

    # A: fronts per seed
    _style(a, "A  Fronts on the fitness views (up and left is better)")
    for run in runs:
        front = sorted(run["front"], key=lambda p: p["kept"])
        x = [100 * p["kept"] for p in front]
        y = [p["quality"] for p in front]
        width = 2.0 if run["method"] == "sweep" else 1.4
        color = COLOR[run["method"]]
        a.step(x, y, where="post", color=color, lw=width, alpha=0.8)
    for method in methods:
        top = max(best_run(runs, method)["front"], key=lambda p: p["quality"])
        x, y = 100 * top["kept"], top["quality"] + 0.6
        a.text(x, y, LABEL[method], color=INK2, fontsize=8.5)
    a.set_xlim(0, 100)
    a.set_xlabel("% of Gaussians kept")
    a.set_ylabel(f"{args['quality']}-view PSNR vs unpruned (dB, clipped 20-60)")

    # B: hypervolume over evaluations
    _style(b, "B  Hypervolume of the front found so far (higher is better)")
    for run in runs:
        curve = run["hypervolume_curve"]
        x = run["check_every"] * (np.arange(len(curve)) + 1)
        b.plot(x, curve, color=COLOR[run["method"]], lw=1.4, alpha=0.8)
    for method in methods:
        mine = [r for r in runs if r["method"] == method]
        end = np.mean([r["hypervolume"] for r in mine])
        x_end = mine[0]["check_every"] * len(mine[0]["hypervolume_curve"])
        label = f" {LABEL[method]} {end:.3f}"
        b.text(x_end, end, label, color=INK2, fontsize=8.5, va="center")
    b.set_xlabel("fitness evaluations")
    b.set_ylabel("hypervolume / box")

    # C: held-out check
    _style(c, "C  Held-out test views, best front per method (dashed: fitness)")
    for method in methods:
        points = sorted(held[method], key=lambda p: p["kept"])
        x = [100 * p["kept"] for p in points]
        color = COLOR[method]
        test = [p["test_worst"] for p in points]
        fit = [p["fit_worst"] for p in points]
        c.plot(x, test, color=color, lw=1.6, marker="o", ms=4)
        c.plot(x, fit, color=color, lw=1.1, ls="--", alpha=0.7)
    c.set_xlim(0, 100)
    c.set_xlabel("% of Gaussians kept")
    c.set_ylabel("worst-view PSNR vs unpruned (dB)")

    # D: MAP-Elites archive
    _style(d, "D  MAP-Elites archive: best quality per cell (dB)")
    archives = [r for r in runs if r["method"] == "map_elites"]
    if archives:
        best = best_run(runs, "map_elites")
        grid = np.array(best["archive"], dtype=float)  # (kept bins, share bins)
        cmap = LinearSegmentedColormap.from_list("aqua", ["#e6f6ef", "#0e6b4a"])
        extent = [0, 100, 0, 100]
        image = d.imshow(
            grid.T, origin="lower", extent=extent, aspect="auto", cmap=cmap
        )
        kept_bins, share_bins = grid.shape
        for i in range(kept_bins):
            for j in range(share_bins):
                if np.isfinite(grid[i, j]):
                    x, y = (i + 0.5) * 100 / kept_bins, (j + 0.5) * 100 / share_bins
                    label = f"{grid[i, j]:.0f}"
                    d.text(x, y, label, ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=d, shrink=0.8, label="quality (dB)")
        d.set_xlabel("% of Gaussians kept")
        d.set_ylabel("% of removals made by the importance gene")
        d.grid(False)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


def publish_front(run_dir, fig_dir=FIG_DIR):
    # front.png and summary.json into the tracked figures/runs/<run>/
    run_dir = Path(run_dir)
    dest = Path(fig_dir) / "runs" / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(plot_front(run_dir), dest / "front.png")
    shutil.copy(run_dir / "summary.json", dest / "summary.json")
    return dest / "front.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="results/<run> directory written by front.py")
    print(publish_front(ap.parse_args().run))


if __name__ == "__main__":
    main()
