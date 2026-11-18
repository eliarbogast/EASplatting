"""strip.py — a render strip per run: what the best genomes of each method actually look like. Needs a GPU.

    python -m easplatting.strip results/<run>        # -> results/<run>/renders.jpg  (experiment.py also calls this after every run)
    python -m easplatting.strip --all                # every results/*/summary.json (no search is rerun; genomes come from summary.json)

Columns: unpruned | best GA genome | best importance-sweep genome | best opacity-sweep genome (those that exist and are feasible).
Rows: two held-out test views, the WORST view of the GA genome (the honest failure case) and the MEDIAN one. For each view three rows:
the pruned renders, an error map (|pruned - unpruned|, same scale everywhere), and a 4x zoom of the window where the GA genome's error
is highest (yellow box, same window in every column). Captions give % kept and PSNR vs the unpruned render.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from easplatting.genome import decode
from easplatting.gs_loader import prune
from easplatting.metrics import psnr_fast
from easplatting.render import render_views
from easplatting.thresholds import keep_mask

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
CAPTION = {"ga": "GA best", "sweep_importance": "importance sweep", "sweep_opacity": "opacity sweep"}
ERR_MAX = 0.25                                           # |difference| that maps to the brightest color
ZOOM = 4                                                 # the zoom window is 1/ZOOM of the image in each dimension


def pick_genomes(runs):
    """{method: run summary} of the feasible run with the fewest Gaussians, for each method that is shown."""
    best = {}
    for r in runs:
        if r["method"] in CAPTION and r["feasible"] and (r["method"] not in best or r["kept"] < best[r["method"]]["kept"]):
            best[r["method"]] = r
    return {m: best[m] for m in CAPTION if m in best}


def worst_window(err, zoom=ZOOM):
    """Top-left (y, x) and size (h, w) of the window with the largest mean error in an (H, W) error map."""
    H, W = err.shape
    h, w = H // zoom, W // zoom
    pooled = F.avg_pool2d(err[None, None], (h, w), stride=(max(h // 8, 1), max(w // 8, 1)))[0, 0]
    iy, ix = divmod(int(pooled.argmax()), pooled.shape[1])
    return iy * max(h // 8, 1), ix * max(w // 8, 1), h, w


@torch.no_grad()
def render_strip(run_dir, ctx=None, out=None):
    from easplatting.experiment import setup_scene
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    args, picks = summary["args"], pick_genomes(summary["runs"])
    if "ga" not in picks:
        return None
    ctx = ctx or setup_scene(args["scene"], args["genes"])
    scene, stats, bounds, cams = ctx["scene"], ctx["stats"], ctx["bounds"], ctx["cams"]
    test = cams.subset(ctx["test_idx"])
    base = render_views(scene, test, batch_size=4)

    def pruned(run):
        return prune(scene, keep_mask(stats, *decode(run["genome"], bounds)))

    ga_psnr = psnr_fast(render_views(pruned(picks["ga"]), test, batch_size=4), base, max_db=60)
    order = ga_psnr.argsort()
    views = [("worst test view", int(order[0])), ("median test view", int(order[len(order) // 2]))]

    cols = [("unpruned", None)] + [(CAPTION[m], r) for m, r in picks.items()]
    n = len(cols)
    fig, axes = plt.subplots(3 * len(views), n, figsize=(3.6 * n, 2.0 * 3 * len(views) + 0.8), facecolor=SURFACE, constrained_layout=True)
    fig.suptitle(f"{args['scene']}: what the best genomes look like ({run_dir.name})", x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    for vi, (vname, v) in enumerate(views):
        ref = base[v]
        imgs = [ref if run is None else render_views(pruned(run), test, [v], batch_size=1)[0] for _, run in cols]
        y, x, h, w = worst_window((imgs[1] - ref).abs().mean(-1))           # window chosen on the GA genome's error
        for ci, ((cname, run), img) in enumerate(zip(cols, imgs)):
            top, err_ax, zoom_ax = axes[3 * vi, ci], axes[3 * vi + 1, ci], axes[3 * vi + 2, ci]
            for a in (top, err_ax, zoom_ax):
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
            if run is None:
                caption = "unpruned (reference)"
                err_ax.text(0.04, 0.5, f"error row: |pruned - unpruned|\nblack = identical\nwhite = off by >= {ERR_MAX:g}\n(of full brightness)\n\nzoom row: {ZOOM}x crop of the\nyellow box (worst spot of\nthe GA genome)", fontsize=8.5,
                            color=INK2, va="center", transform=err_ax.transAxes)
            else:
                p = psnr_fast(img[None], ref[None], max_db=60).item()
                caption = f"{cname}: {100 * run['kept']:.1f}% kept, {p:.1f} dB"
                err = (img - ref).abs().mean(-1).clamp(0, ERR_MAX) / ERR_MAX
                err_ax.imshow(err.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            top.imshow(img.clamp(0, 1).cpu().numpy())
            top.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#f5d90a", linewidth=1.6))
            top.set_title(caption, fontsize=9, color=INK, loc="left")
            zoom_ax.imshow(img[y:y + h, x:x + w].clamp(0, 1).cpu().numpy())
            if ci == 0:
                top.set_ylabel(f"{vname}\n#{Path(str(test.names[v])).stem}", fontsize=8.5, color=INK2)
    out = Path(out or run_dir / "renders.jpg")
    fig.savefig(out, dpi=100, facecolor=SURFACE, pil_kwargs=dict(quality=85))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="results/<run> directory")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    runs = sorted(p.parent for p in Path("results").glob("*/summary.json")) if a.all else [Path(a.run)]
    from easplatting.experiment import setup_scene
    ctxs = {}
    for r in runs:
        args = json.loads((r / "summary.json").read_text())["args"]
        key = (args["scene"], args["genes"])
        if key not in ctxs:
            ctxs.clear()                                  # one scene on the GPU at a time
            ctxs[key] = setup_scene(*key)
        print(render_strip(r, ctxs[key]), flush=True)


if __name__ == "__main__":
    main()
