"""
measure.py — measure things on THIS machine that feed the numbers in docs/EA_PLAN.md.

    python scripts/measure.py cost       # ~2 min: what does one EA fitness evaluation cost on this GPU?
    python scripts/measure.py metrics    # ~3 min: what do PSNR / SSIM / LPIPS mean for pruning?

Run from anywhere with the env active and a GPU visible; needs the bonsai data from get_data.sh
(`cost` also adds a truck row if scripts/train_gsplat.sh has produced results/truck_gsplat_7k).

cost     An evaluation = threshold mask -> prune -> render 8 train views at half resolution -> PSNR vs cached
         baselines. Reports (A) cost vs scene size, incl. synthetic 5M/10M-Gaussian scenes made by tiling bonsai
         (a rough proxy for the outdoor Mip-NeRF 360 scenes), (B) whether one process already saturates the GPU
         (aggregate evals/s with 1, 2, 4 processes), (C) the extra cost of SSIM / LPIPS inside the loop.
         Re-run it on each new GPU type and update the compute budget in docs/EA_PLAN.md.

metrics  On bonsai (8 train views, full res): (1) sweeps an opacity threshold and renders what each quality level
         looks like, (2) samples 88 genomes and checks how the metrics rank them and rank the REAL loss vs the
         photos, (3) tabulates what a PSNR-vs-unpruned level means as loss vs the photos, (4) checks that the cheap
         half-resolution fitness ranks like the full-resolution one. Writes results/metrics_intuition/
         {levels.png, frontier.png, numbers.json}.
"""

import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from easplatting.cameras import Cameras, evenly_spaced, load_cameras, load_image, split_indices  # noqa: E402
from easplatting.gs_loader import activate, load_scene, prune  # noqa: E402
from easplatting.metrics import ImageMetrics, psnr_fast  # noqa: E402
from easplatting.render import render_views  # noqa: E402
from easplatting.thresholds import gene_stats, keep_mask, slider_bounds  # noqa: E402

DEV = "cuda"
BONSAI = ("pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply", "data/360_v2/bonsai", "images_2")
TRUCK = ("results/truck_gsplat_7k/ckpts/ckpt_6999_rank0.pt", "results/truck_gsplat_7k/cameras.json", None)
INF = float("inf")


# ------------------------------------------------------------------------------------ helpers


def scaled(cams, s):
    """The same cameras at image scale s (intrinsics and size)."""
    Ks = cams.Ks.clone()
    Ks[:, :2, :] *= s
    return Cameras(names=cams.names, viewmats=cams.viewmats, Ks=Ks,
                   widths=[int(w * s) for w in cams.widths], heights=[int(h * s) for h in cams.heights])


def timed(fn, n=20):
    """Mean milliseconds per call, GPU-synchronized, after one warm-up."""
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def load_scene_and_views(spec, copies=1, scale=0.5, n_views=8):
    """Activated scene (optionally tiled `copies` times with jitter) and 8 evenly spaced train views."""
    act = activate(load_scene(spec[0], device=DEV))
    if copies > 1:   # synthetic bigger scene: jittered copies (overdraw-heavy, a rough proxy for denser scenes)
        g = torch.Generator(device=DEV).manual_seed(0)
        parts = []
        for _ in range(copies):
            p = dict(act)
            p["means"] = act["means"] + 0.01 * torch.randn(act["means"].shape, device=DEV, generator=g)
            parts.append(p)
        act = {k: (torch.cat([p[k] for p in parts]) if torch.is_tensor(act[k]) and act[k].ndim > 0 else act[k])
               for k in act}
    cams = load_cameras(spec[1], images=spec[2]) if spec[2] else load_cameras(spec[1])
    train, _ = split_indices(len(cams), 8)
    views = cams.subset(evenly_spaced(train, n_views))
    return act, views, scaled(views, scale)


# ------------------------------------------------------------------------------------- cost


def _worker():
    """One process of the GPU-sharing test: prints evals/s for a realistic genome."""
    act, _, fc = load_scene_and_views(BONSAI)
    stats = gene_stats(act)
    base = render_views(act, fc, batch_size=8)
    th = (0.01, 10 ** 0.4, 10 ** 3.0)

    def ev():
        sp = prune(act, keep_mask(stats, *th))
        return float(psnr_fast(render_views(sp, fc, batch_size=8), base).mean())

    ev()
    torch.cuda.synchronize()
    t = time.perf_counter()
    n = 150
    for _ in range(n):
        ev()
    torch.cuda.synchronize()
    print(f"{n / (time.perf_counter() - t):.1f}")


def cost():
    print("== A. cost vs scene size (K=8 views @ half res, unpruned scene; ms)")
    print(f"{'scene':<28} {'N':>11} {'render':>8} {'prune-copy':>10} {'zero-opacity eval':>18} {'peak GB':>8}")
    rows = [("bonsai (NB 3DGS)", BONSAI, 1), ("truck (gsplat 7k)", TRUCK, 1),
            ("bonsai x4 (synthetic)", BONSAI, 4), ("bonsai x8 (synthetic)", BONSAI, 8)]
    for label, spec, copies in rows:
        if not Path(spec[0]).exists():
            print(f"(skipping {label}: {spec[0]} not found)")
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        act, _, fc = load_scene_and_views(spec, copies)
        n = act["means"].shape[0]
        mask = keep_mask(gene_stats(act), 0.0, INF, INF)           # keep everything = worst case for render time

        def zero():
            sp = dict(act)
            sp["opacities"] = act["opacities"] * mask
            return render_views(sp, fc, batch_size=8)

        r = timed(lambda: render_views(act, fc, batch_size=8))
        p = timed(lambda: prune(act, mask))
        z = timed(zero)
        print(f"{label:<28} {n:>11,} {r:8.1f} {p:10.1f} {z:18.1f} {torch.cuda.max_memory_allocated() / 1e9:8.2f}")
        del act, mask
        torch.cuda.empty_cache()

    print("\n== B. does the GPU saturate? aggregate evals/s with P identical processes (bonsai, K=8 @ half res)")
    for procs_n in (1, 2, 4):
        procs = [subprocess.Popen([sys.executable, __file__, "_worker"], stdout=subprocess.PIPE, text=True)
                 for _ in range(procs_n)]
        rates = [float(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
        print(f"  {procs_n} process(es): per-process {st.mean(rates):5.1f} evals/s, aggregate {sum(rates):5.1f} evals/s")

    print("\n== C. cost of richer metrics inside the loop (8 views @ half res; ms per evaluation)")
    act, _, fc = load_scene_and_views(BONSAI)
    base = render_views(act, fc, batch_size=8)
    sp = prune(act, keep_mask(gene_stats(act), 0.01, 10 ** 0.4, 10 ** 3.0))
    imgs = render_views(sp, fc, batch_size=8)
    m_vgg, m_alex = ImageMetrics(device=DEV, lpips_net="vgg"), ImageMetrics(device=DEV, lpips_net="alex")
    lpips_extra = lambda m: timed(lambda: m(imgs, base, lpips=True), 3) - timed(lambda: m(imgs, base, lpips=False), 3)
    print(f"  render only                : {timed(lambda: render_views(sp, fc, batch_size=8)):6.1f}")
    print(f"  + PSNR (psnr_fast)         : {timed(lambda: psnr_fast(imgs, base)):6.2f}   (extra)")
    print(f"  + SSIM+PSNR (torchmetrics) : {timed(lambda: m_vgg(imgs, base, lpips=False), 5):6.1f}   (extra)")
    print(f"  + LPIPS-alex               : {lpips_extra(m_alex):6.1f}   (extra)")
    print(f"  + LPIPS-vgg                : {lpips_extra(m_vgg):6.1f}   (extra)")


# ----------------------------------------------------------------------------------- metrics


def metrics():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    out = ROOT / "results" / "metrics_intuition"
    out.mkdir(parents=True, exist_ok=True)
    act = activate(load_scene(BONSAI[0], device=DEV))
    n_total = act["means"].shape[0]
    stats = gene_stats(act)
    b = slider_bounds(stats)
    cams = load_cameras(BONSAI[1], images=BONSAI[2])
    train, _ = split_indices(len(cams), 8)
    sub = cams.subset(evenly_spaced(train, 8))
    W, H = sub.size
    gt = torch.stack([load_image(sub.image_paths[j], (0, 0, 0), (W, H)) for j in range(8)]).to(DEV)
    base = render_views(act, sub, batch_size=8)
    half = scaled(sub, 0.5)
    base_half = render_views(act, half, batch_size=8)
    m_vgg, m_alex = ImageMetrics(DEV, "vgg"), ImageMetrics(DEV, "alex")

    def all_metrics(pred, ref):
        v, a = m_vgg(pred, ref), m_alex(pred, ref)
        mean = lambda x: float(np.mean(x))
        return dict(psnr=mean([min(p, 99.0) for p in v["psnr"]]), ssim=mean(v["ssim"]),
                    lpips_vgg=mean(v["lpips"]), lpips_alex=mean(a["lpips"]))

    g0 = all_metrics(base, gt)
    print(f"unpruned vs GT (8 train views, full res): PSNR {g0['psnr']:.2f}  SSIM {g0['ssim']:.4f}  "
          f"LPIPS-alex {g0['lpips_alex']:.4f}  LPIPS-vgg {g0['lpips_vgg']:.4f}")

    def run(th):
        """Metrics of one genome vs the unpruned render (U) and vs the photos (G); dG = real loss."""
        mask = keep_mask(stats, *th)
        sp = prune(act, mask)
        cheap = float(psnr_fast(render_views(sp, half, batch_size=8), base_half, max_db=99).mean())
        imgs = render_views(sp, sub, batch_size=8)
        u, g = all_metrics(imgs, base), all_metrics(imgs, gt)
        dg = dict(psnr=g0["psnr"] - g["psnr"], ssim=g0["ssim"] - g["ssim"],
                  lpips_vgg=g["lpips_vgg"] - g0["lpips_vgg"], lpips_alex=g["lpips_alex"] - g0["lpips_alex"])
        return dict(th=th, kept=int(mask.sum()) / n_total, U=u, G=g, cheap=cheap, dG=dg), imgs

    # (1) opacity-only sweep (other genes truly off) -> quality-size curve and the pictures
    sweep, sweep_imgs = [], {}
    for o in np.logspace(-3, math.log10(0.6), 28):
        r, imgs = run((float(o), INF, INF))
        sweep.append(r)
        sweep_imgs[float(o)] = imgs[1].cpu()
    # (2) 60 random genomes where an EA would search; each gene is "off" with probability 0.25
    rng = np.random.default_rng(0)
    rnd = []
    for _ in range(60):
        o = 10 ** rng.uniform(-3, math.log10(0.3))
        s = 10 ** rng.uniform(sum(b["max_scale"]) / 2, b["max_scale"][1])
        a = 10 ** rng.uniform(sum(b["max_anisotropy"]) / 2, b["max_anisotropy"][1])
        o = 0.0 if rng.random() < 0.25 else o
        s = INF if rng.random() < 0.25 else s
        a = INF if rng.random() < 0.25 else a
        rnd.append(run((float(o), float(s), float(a)))[0])
    json.dump(dict(unpruned_vs_gt=g0, sweep=sweep, random=rnd), open(out / "numbers.json", "w"), default=float)
    rows = sweep + rnd

    print("\nvs-unpruned PSNR band | n | mean kept | loss vs photos: dPSNR(dB)  dSSIM    dLPIPS-alex  dLPIPS-vgg")
    for lo, hi in [(60, 100), (50, 60), (45, 50), (40, 45), (35, 40), (30, 35), (0, 30)]:
        sel = [r for r in rows if lo <= r["U"]["psnr"] < hi]
        if sel:
            mk = lambda f: np.mean([f(r) for r in sel])
            print(f"  {lo:>2}-{hi:<3} dB          | {len(sel):>2} | {mk(lambda r: r['kept']):.2f}      | "
                  f"{mk(lambda r: r['dG']['psnr']):9.3f}  {mk(lambda r: r['dG']['ssim']):8.4f}  "
                  f"{mk(lambda r: r['dG']['lpips_alex']):10.4f}  {mk(lambda r: r['dG']['lpips_vgg']):10.4f}")

    keys = [("PSNR", lambda r: r["U"]["psnr"]), ("SSIM", lambda r: r["U"]["ssim"]),
            ("-LPIPS-alex", lambda r: -r["U"]["lpips_alex"]), ("-LPIPS-vgg", lambda r: -r["U"]["lpips_vgg"]),
            ("kept (neg.)", lambda r: -r["kept"])]
    rho = lambda f1, f2: spearmanr([f1(r) for r in rows], [f2(r) for r in rows])[0]
    print(f"\nSpearman rank agreement of metrics vs-unpruned across {len(rows)} genomes (1.0 = identical ranking):")
    print("             " + "".join(f"{k:>12}" for k, _ in keys))
    for k1, f1 in keys:
        print(f"{k1:>12} " + "".join(f"{rho(f1, f2):12.3f}" for _, f2 in keys))
    print("\nand how well does each vs-unpruned metric predict the REAL loss vs the photos (rank corr with -dPSNR_GT):")
    for k, f in keys[:4]:
        print(f"  {k:>12}: {rho(f, lambda r: -r['dG']['psnr']):.3f}")
    print("\ncheap fitness (half res, 8 views) vs full-res PSNR on the same views: Spearman %.3f ; vs REAL loss vs photos: %.3f"
          % (rho(lambda r: r["cheap"], lambda r: r["U"]["psnr"]), rho(lambda r: r["cheap"], lambda r: -r["dG"]["psnr"])))

    # figure 1: what do the quality levels look like? (crop = the region the harshest prune damages most)
    picks = [min(sweep, key=lambda r: abs(r["U"]["psnr"] - t)) for t in (50, 40, 35, 30, 25)]
    ref = base[1].cpu()
    ch, cw = 320, 480
    err = (sweep_imgs[picks[-1]["th"][0]] - ref).abs().mean(-1)
    pooled = torch.nn.functional.avg_pool2d(err[None, None], (ch, cw), stride=(40, 40))[0, 0]
    iy, ix = np.unravel_index(int(pooled.argmax()), pooled.shape)
    y0, x0 = iy * 40, ix * 40
    fig, ax = plt.subplots(2, 6, figsize=(19, 6.6))
    show = lambda a, im, title: (a.imshow(im[y0:y0 + ch, x0:x0 + cw].clamp(0, 1).numpy()), a.set_title(title, fontsize=9), a.axis("off"))
    show(ax[0, 0], ref, "unpruned (reference)\nkept 100%")
    ax[1, 0].axis("off")
    ax[1, 0].text(0.02, 0.5, "bottom row = |pruned - unpruned| x 8\n(brighter = more different)", fontsize=9, transform=ax[1, 0].transAxes)
    for j, r in enumerate(picks, 1):
        im = sweep_imgs[r["th"][0]]
        show(ax[0, j], im, f"opacity >= {r['th'][0]:.3f}: kept {100 * r['kept']:.0f}%\nPSNR {r['U']['psnr']:.1f} dB  "
                           f"SSIM {r['U']['ssim']:.3f}\nLPIPS-alex {r['U']['lpips_alex']:.3f}  (vs unpruned)")
        show(ax[1, j], ((im - ref).abs() * 8).clamp(0, 1), f"loss vs photos: {r['dG']['psnr']:.2f} dB")
    fig.tight_layout()
    fig.savefig(out / "levels.png", dpi=80)
    plt.close(fig)

    # figure 2: quality-size curve of one gene, and what "vs unpruned" means for real quality
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot([100 * r["kept"] for r in sweep], [r["U"]["psnr"] for r in sweep], "o-")
    for t in (40, 35, 30):
        ax[0].axhline(t, color="gray", ls=":", lw=0.8)
    ax[0].set(xlabel="Gaussians kept (%)  [opacity threshold only]", ylabel="PSNR vs unpruned render (dB)",
              title="Quality-size curve for one gene (bonsai)")
    ax[0].invert_xaxis()
    ax[1].scatter([r["U"]["psnr"] for r in rows], [r["dG"]["psnr"] for r in rows], s=14)
    ax[1].set(xlabel="PSNR vs unpruned render (dB)", ylabel="actual loss vs the photos (dB of PSNR)", yscale="log",
              title="What 'vs unpruned' means for real quality")
    fig.tight_layout()
    fig.savefig(out / "frontier.png", dpi=80)
    print(f"\nsaved figures and numbers to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["cost", "metrics", "_worker"], help="_worker is internal (GPU-sharing test)")
    args = ap.parse_args()
    {"cost": cost, "metrics": metrics, "_worker": _worker}[args.what]()


if __name__ == "__main__":
    main()
