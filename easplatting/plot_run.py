"""plot_run.py — one standard PNG per experiment run, and one comparison PNG across runs. CPU only.

    python -m easplatting.plot_run results/bonsai_4g_v32_w35   # -> results/bonsai_4g_v32_w35/plots.png
    python -m easplatting.plot_run --all                       # every results/*/summary.json, plus figures/compare.png

plots.png (reads summary.json + log.jsonl written by experiment.py):
  A  best feasible % kept so far vs evaluations (GA / random per seed; sweeps as dashed lines)
  B  every evaluation in the (% kept, mean PSNR) plane; the constraint is the dashed line
  C  best % kept per run          D  the same genomes re-checked: fitness views, all train views, test views, worst view
compare.png: % kept per method, one panel per scene, one row per fitness setting (views / constraint).
experiment.py calls publish() after every run: plots.png, summary.json and renders.jpg (see strip.py) are copied into the
TRACKED folder figures/runs/<run>/ (results/ is git-ignored) and figures/compare.png is refreshed. Commit figures/ with the run.
images/ (full-resolution lossless renders, ~17 MB per run) is only copied with --publish-images (experiment.py) or --images (here).
Colors: GA blue, random orange, importance sweep aqua, other sweeps gray (validated slots 1-3 of the dataviz palette).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
COLOR = {"ga": BLUE, "random": ORANGE, "sweep_importance": AQUA}
LABEL = {"ga": "GA", "random": "random", "sweep_importance": "importance sweep", "sweep_opacity": "opacity sweep",
         "sweep_scale": "scale sweep", "sweep_anisotropy": "anisotropy sweep"}
ORDER = list(LABEL)


def _color(m):
    return COLOR.get(m, GRAY)


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def load_run(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines() if l.strip()]
    return summary["args"], summary["runs"], rows


def setting_label(args):
    views, worst = args.get("views", 8), args.get("worst_tau")
    return f"{views} views, mean >= {args['tau']:g} dB" + (f", worst >= {worst:g} dB" if worst else "")


def best_so_far(rows):
    """Running min of % kept among feasible evaluations (nan until the first feasible one)."""
    out, best = [], np.nan
    for r in rows:
        if r["fitness"] >= 0:
            best = r["kept"] * 100 if np.isnan(best) else min(best, r["kept"] * 100)
        out.append(best)
    return np.array(out)


def plot_run(run_dir, out=None):
    run_dir = Path(run_dir)
    args, runs, rows = load_run(run_dir)
    out = Path(out or run_dir / "plots.png")
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.6), facecolor=SURFACE, constrained_layout=True)
    (a, b), (c, d) = ax
    fig.suptitle(f"{args['scene']}, {args['genes']} genes: {setting_label(args)}   ({run_dir.name})", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    by = {}
    for r in rows:
        by.setdefault((r["method"], r["seed"]), []).append(r)

    # A: convergence
    _style(a, "A  Best feasible % kept vs evaluations (lower is better)")
    for (m, s), rs in by.items():
        y = best_so_far(rs)
        if m in ("ga", "random"):
            a.plot(np.arange(len(y)), y, color=_color(m), lw=1.6, alpha=0.9)
        elif m in ("sweep_importance", "sweep_opacity") and not np.isnan(y[-1]):
            a.axhline(y[-1], color=_color(m), lw=1.2, ls="--")
            a.text(1, y[-1] + 0.8, f"{LABEL[m]} {y[-1]:.1f}", color=INK2, fontsize=8, ha="right", va="bottom", transform=a.get_yaxis_transform())
    for m in ("ga", "random"):
        if any(k[0] == m for k in by):
            ys = [best_so_far(rs) for (mm, _), rs in by.items() if mm == m]
            end = np.nanmean([y[-1] for y in ys])
            a.text(len(ys[0]) - 1, end - 1.5, f"{LABEL[m]} {end:.1f}", color=_color(m), fontsize=9, ha="right", va="top", fontweight="bold")
    a.set_ylim(top=100)
    a.set_xlabel("fitness evaluations")
    a.set_ylabel("% of Gaussians kept")

    # B: size-quality cloud
    _style(b, "B  Every evaluation: % kept vs mean PSNR on the fitness views")
    for m in ("random", "ga"):
        pts = np.array([[r["kept"] * 100, r["psnr"]] for (mm, _), rs in by.items() if mm == m for r in rs if r["psnr"] > 0])
        if len(pts):
            b.scatter(pts[:, 0], np.minimum(pts[:, 1], 60), s=9, color=_color(m), alpha=0.35, linewidths=0, label=LABEL[m])
    for m in ("sweep_opacity", "sweep_importance"):
        pts = np.array(sorted([r["kept"] * 100, min(r["psnr"], 60)] for (mm, _), rs in by.items() if mm == m for r in rs if r["psnr"] > 0))
        if len(pts):
            b.plot(pts[:, 0], pts[:, 1], color=_color(m), lw=1.8, label=LABEL[m], zorder=3)
    b.axhline(args["tau"], color=INK2, ls="--", lw=1)
    b.text(0.01, args["tau"] + 0.5, f"constraint {args['tau']:g} dB", color=INK2, fontsize=8, ha="left", transform=b.get_yaxis_transform())
    b.set_ylim(25, 61)
    b.set_xlabel("% of Gaussians kept")
    b.set_ylabel("mean PSNR vs unpruned (dB, capped 60)")
    b.legend(fontsize=8, frameon=False, loc="upper left")

    # C / D: best genome of each run, same rows
    runs = sorted(runs, key=lambda r: (ORDER.index(r["method"]) if r["method"] in ORDER else 99, r["seed"]))
    runs = [r for r in runs if r["method"] in ("ga", "random", "sweep_importance", "sweep_opacity")] or runs
    names = [f"{LABEL.get(r['method'], r['method'])}" + (f" s{r['seed']}" if r["method"] in ("ga", "random") else "") for r in runs]
    y = np.arange(len(runs))[::-1]
    _style(c, "C  Best genome of each run: % kept")
    c.barh(y, [100 * r["kept"] for r in runs], color=[_color(r["method"]) for r in runs], height=0.62)
    for yi, r in zip(y, runs):
        c.text(100 * r["kept"] + 0.8, yi, f"{100 * r['kept']:.1f}", va="center", fontsize=8, color=INK)
    c.set_yticks(y, names)
    c.set_xlim(0, 105)
    c.set_xlabel("% of Gaussians kept")
    _style(d, "D  Same genomes, re-measured (dB vs unpruned)")
    d.set_yticks(y, names)
    d.tick_params(labelleft=False)
    series = [("fit_psnr", "o", "fitness views (mean)"), ("train_psnr", "s", "all train views (mean)"),
              ("test_psnr", "D", "test views (mean)"), ("test_worst", "X", "test views (worst)")]
    for key, mk, lab in series:
        pts = [(min(r[key], 60), yi) for yi, r in zip(y, runs) if key in r and r.get("feasible", True)]
        if pts:
            d.scatter(*zip(*pts), marker=mk, s=34, edgecolors=INK, facecolors=INK if key != "train_psnr" else "none", linewidths=1.2, label=lab, zorder=3)
    d.axvline(args["tau"], color=INK2, ls="--", lw=1)
    if args.get("worst_tau"):
        d.axvline(args["worst_tau"], color=INK2, ls=":", lw=1)
        d.text(args["worst_tau"] - 0.4, 0.99, "worst-view floor", fontsize=7.5, color=INK2, ha="right", va="top", transform=d.get_xaxis_transform())
    d.text(args["tau"] + 0.4, 0.99, "mean constraint", fontsize=7.5, color=INK2, ha="left", va="top", transform=d.get_xaxis_transform())
    d.set_xlim(15, 62)
    c.set_ylim(-0.7, len(runs) - 0.3)
    d.set_ylim(-0.7, len(runs) - 0.3)
    d.set_xlabel("PSNR vs unpruned render (dB, capped 60)")
    d.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


FIG_DIR = Path("figures")


def plot_compare(results_dir="results", out=None):
    results_dir = Path(results_dir)
    out = Path(out or FIG_DIR / "compare.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cells = {}                                   # (setting, scene) -> {method: [runs]}
    for p in sorted(results_dir.glob("*/summary.json")):
        s = json.loads(p.read_text())
        key = (setting_label(s["args"]), s["args"]["scene"])
        if s["args"].get("genes") != 4:
            continue
        cells.setdefault(key, {})
        for r in s["runs"]:
            if r["feasible"]:
                cells[key].setdefault(r["method"], []).append(r)
    scenes = sorted({k[1] for k in cells}, key=lambda n: (["bonsai", "counter", "garden", "bicycle"] + [n]).index(n))
    multi = {k[0] for k in cells if sum(1 for q in cells if q[0] == k[0]) >= 2}   # skip one-off diagnostic settings
    settings = sorted(multi or {k[0] for k in cells}, key=lambda t: (int(t.split()[0]), "worst" in t))
    if not settings:
        return None
    show = ["ga", "random", "sweep_importance", "sweep_opacity"]
    fig, axes = plt.subplots(len(settings), len(scenes), figsize=(3.3 * len(scenes) + 1, 2.9 * len(settings) + 0.8),
                             facecolor=SURFACE, sharey=True, squeeze=False, constrained_layout=True)
    fig.suptitle("% of Gaussians kept at the quality constraint, 4 genes (lower is better; GA/random: mean of seeds, bar = range)",
                 x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    for i, st in enumerate(settings):
        for j, sc in enumerate(scenes):
            ax = axes[i, j]
            _style(ax, f"{sc}" if i == 0 else "")
            ax.set_ylim(0, 112)
            runs = cells.get((st, sc))
            if j == 0:
                ax.set_ylabel(st.replace(", ", "\n"), fontsize=8, color=INK)
            if not runs:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", color=INK2, fontsize=9, transform=ax.transAxes)
                ax.set_xticks([])
                continue
            for x, m in enumerate(show):
                rs = runs.get(m)
                if not rs:
                    continue
                k = np.array([100 * r["kept"] for r in rs])
                ax.bar(x, k.mean(), color=_color(m), width=0.66)
                if len(k) > 1:
                    ax.plot([x, x], [k.min(), k.max()], color=INK, lw=1.2)
                ax.text(x, k.max() + 2, f"{k.mean():.0f}", ha="center", fontsize=8, color=INK)
                w = [r["test_worst"] for r in rs if "test_worst" in r]
                if w:
                    ax.text(x, 3, f"{np.mean(w):.0f}", ha="center", fontsize=7.5, color="white")
            ax.set_xticks(range(len(show)), ["GA", "rand", "imp.", "opac."], fontsize=8)
    fig.text(0.01, -0.01, "white number inside a bar = mean worst-test-view PSNR (dB) of that method's best genomes, when measured",
             fontsize=7.5, color=INK2)
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def publish(run_dir, fig_dir=FIG_DIR, images=False):
    """Copy a run's plots.png, summary.json, renders.jpg into the tracked figures/runs/<run>/ and refresh compare.png.
    images=True also copies the full-resolution images/ (~17 MB per run, so opt-in: python -m easplatting.plot_run <run> --images)."""
    run_dir = Path(run_dir)
    dest = Path(fig_dir) / "runs" / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(plot_run(run_dir), dest / "plots.png")
    shutil.copy(run_dir / "summary.json", dest / "summary.json")
    if (run_dir / "renders.jpg").exists():
        shutil.copy(run_dir / "renders.jpg", dest / "renders.jpg")
    if images and (run_dir / "images").exists():
        shutil.copytree(run_dir / "images", dest / "images", dirs_exist_ok=True)
    return dest / "plots.png", plot_compare(run_dir.parent, Path(fig_dir) / "compare.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="results/<run> directory")
    ap.add_argument("--all", action="store_true", help="every run under results/, plus figures/compare.png")
    ap.add_argument("--images", action="store_true", help="also publish the full-resolution images/ of the given run")
    args = ap.parse_args()
    if args.run:
        print(*publish(args.run, images=args.images))
    if args.all:
        for p in sorted(Path("results").glob("*/summary.json")):
            publish(p.parent)
            print(p.parent.name)
        print(plot_compare())


if __name__ == "__main__":
    main()
