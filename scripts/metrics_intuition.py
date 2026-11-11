"""
metrics_intuition.py — what do PSNR / SSIM / LPIPS vs the unpruned render actually mean for pruning?

    python scripts/metrics_intuition.py           # repo root, env active, GPU visible (~3 min), needs bonsai data

On bonsai (8 train views, full res) it (1) sweeps an opacity threshold and renders what each quality level
looks like, (2) samples 88 genomes and checks how the metrics rank them (and rank the REAL loss vs the
photos), (3) tabulates what a PSNR-vs-unpruned level means as loss vs the photos, and (4) checks that the
cheap half-resolution fitness ranks like the full-resolution one. Writes results/metrics_intuition/
{levels.png, frontier.png, numbers.json}. Findings are summarized in docs/EA_PLAN.md.
"""
import os, sys, math, json
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from cameras import load_cameras, load_image, split_indices, evenly_spaced
from gs_loader import load_scene, activate, prune
from render import render_views
from metrics import ImageMetrics
from viewer import gene_stats, slider_bounds, keep_mask
OUT = str(ROOT / "results" / "metrics_intuition"); os.makedirs(OUT, exist_ok=True)
dev = "cuda"
act = activate(load_scene("pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply", device=dev))
N = act["means"].shape[0]; stats = gene_stats(act); b = slider_bounds(stats)
cams = load_cameras("data/360_v2/bonsai", images="images_2"); train, _ = split_indices(len(cams), 8)
sub = cams.subset(evenly_spaced(train, 8)); W, H = sub.size
gt = torch.stack([load_image(sub.image_paths[j], (0, 0, 0), (W, H)) for j in range(8)]).to(dev)
base = render_views(act, sub, batch_size=8)
mv, ma = ImageMetrics(dev, "vgg"), ImageMetrics(dev, "alex")

def allm(pred, ref):
    v = mv(pred, ref); a = ma(pred, ref)
    mean = lambda x: float(np.mean(x))
    return dict(psnr=mean([min(p, 99.0) for p in v["psnr"]]), ssim=mean(v["ssim"]), lpips_vgg=mean(v["lpips"]), lpips_alex=mean(a["lpips"]))

g0 = allm(base, gt)
print(f"unpruned vs GT (8 train views, full res): PSNR {g0['psnr']:.2f}  SSIM {g0['ssim']:.4f}  LPIPS-alex {g0['lpips_alex']:.4f}  LPIPS-vgg {g0['lpips_vgg']:.4f}")

from metrics import psnr_fast
from cameras import Cameras
Ks_h = sub.Ks.clone(); Ks_h[:, :2, :] *= 0.5
half = Cameras(names=sub.names, viewmats=sub.viewmats, Ks=Ks_h, widths=[int(w / 2) for w in sub.widths], heights=[int(h / 2) for h in sub.heights])
base_half = render_views(act, half, batch_size=8)
def run(th):
    mask = keep_mask(stats, *th); kept = int(mask.sum()) / N
    sp = prune(act, mask)
    cheap = float(psnr_fast(render_views(sp, half, batch_size=8), base_half, max_db=99).mean())
    imgs = render_views(sp, sub, batch_size=8)
    u, g = allm(imgs, base), allm(imgs, gt)
    return dict(th=th, kept=kept, U=u, G=g, cheap=cheap, dG=dict(psnr=g0["psnr"] - g["psnr"], ssim=g0["ssim"] - g["ssim"],
                lpips_vgg=g["lpips_vgg"] - g0["lpips_vgg"], lpips_alex=g["lpips_alex"] - g0["lpips_alex"])), imgs

off_s, off_a = float("inf"), float("inf")   # TRUE off (the aniso slider top end is the 99.9th pct, not off)
# (1) opacity-only sweep -> frontier + the images for the visual
sweep = []; keep_imgs = {}
for o in np.logspace(-3, math.log10(0.6), 28):
    r, imgs = run((float(o), off_s, off_a)); sweep.append(r); keep_imgs[float(o)] = imgs[1].cpu()
# (2) 60 random genomes in the region an EA would actually search (all three genes active)
rng = np.random.default_rng(0); rnd = []
for _ in range(60):
    o = 10 ** rng.uniform(-3, math.log10(0.3)); s = 10 ** rng.uniform(*(sum(b["max_scale"]) / 2, b["max_scale"][1]))
    a = 10 ** rng.uniform(sum(b["max_anisotropy"]) / 2, b["max_anisotropy"][1])
    o = 0.0 if rng.random() < 0.25 else o; s = float("inf") if rng.random() < 0.25 else s; a = float("inf") if rng.random() < 0.25 else a
    rnd.append(run((float(o), float(s), float(a)))[0])
json.dump(dict(unpruned_vs_gt=g0, sweep=sweep, random=rnd), open(f"{OUT}/numbers.json", "w"), default=float)

# ---- table: what does a PSNR-vs-unpruned level mean in loss vs the photos?
allr = sweep + rnd
bands = [(60, 100), (50, 60), (45, 50), (40, 45), (35, 40), (30, 35), (0, 30)]
print("\nvs-unpruned PSNR band | n | mean kept | loss vs photos: dPSNR(dB)  dSSIM    dLPIPS-alex  dLPIPS-vgg")
for lo_, hi_ in bands:
    sel = [r for r in allr if lo_ <= r["U"]["psnr"] < hi_]
    if sel:
        mk = lambda f: np.mean([f(r) for r in sel])
        print(f"  {lo_:>2}-{hi_:<3} dB          | {len(sel):>2} | {mk(lambda r: r['kept']):.2f}      | {mk(lambda r: r['dG']['psnr']):9.3f}  {mk(lambda r: r['dG']['ssim']):8.4f}  {mk(lambda r: r['dG']['lpips_alex']):10.4f}  {mk(lambda r: r['dG']['lpips_vgg']):10.4f}")

# ---- do the metrics agree about which genome is better? (Spearman over all 88 genomes, vs-unpruned)
keys = [("PSNR", lambda r: r["U"]["psnr"]), ("SSIM", lambda r: r["U"]["ssim"]), ("-LPIPS-alex", lambda r: -r["U"]["lpips_alex"]),
        ("-LPIPS-vgg", lambda r: -r["U"]["lpips_vgg"]), ("kept (neg.)", lambda r: -r["kept"])]
print("\nSpearman rank agreement of metrics vs-unpruned across %d genomes (1.0 = identical ranking):" % len(allr))
print("             " + "".join(f"{k:>12}" for k, _ in keys))
for k1, f1 in keys:
    print(f"{k1:>12} " + "".join(f"{spearmanr([f1(r) for r in allr], [f2(r) for r in allr])[0]:12.3f}" for _, f2 in keys))
print("\nand how well does each vs-unpruned metric predict the REAL loss vs the photos (rank corr with -dPSNR_GT):")
for k, f in keys[:4]:
    print(f"  {k:>12}: {spearmanr([f(r) for r in allr], [-r['dG']['psnr'] for r in allr])[0]:.3f}")

# ---- figure 1: what do the levels look like?
targets = [50, 40, 35, 30, 25]; picks = []
for t in targets:
    r = min(sweep, key=lambda r: abs(r["U"]["psnr"] - t)); picks.append(r)
ref = base[1].cpu(); ch, cw = 320, 480
err_all = (keep_imgs[picks[-1]["th"][0]] - ref).abs().mean(-1)
pooled = torch.nn.functional.avg_pool2d(err_all[None, None], (ch, cw), stride=(40, 40))[0, 0]
iy, ix = np.unravel_index(int(pooled.argmax()), pooled.shape); y0, x0 = iy * 40, ix * 40
fig, ax = plt.subplots(2, 6, figsize=(19, 6.6)); ax = np.array(ax)
def show(a, im, title): a.imshow(im[y0:y0 + ch, x0:x0 + cw].clamp(0, 1).numpy()); a.set_title(title, fontsize=9); a.axis("off")
show(ax[0, 0], ref, f"unpruned (reference)\nkept 100%"); ax[1, 0].axis("off")
ax[1, 0].text(0.02, 0.5, "bottom row = |pruned - unpruned| x 8\n(brighter = more different)", fontsize=9, transform=ax[1, 0].transAxes)
for j, r in enumerate(picks, 1):
    im = keep_imgs[r["th"][0]]
    show(ax[0, j], im, f"opacity >= {r['th'][0]:.3f}: kept {100 * r['kept']:.0f}%\nPSNR {r['U']['psnr']:.1f} dB  SSIM {r['U']['ssim']:.3f}\nLPIPS-alex {r['U']['lpips_alex']:.3f}  (vs unpruned)")
    show(ax[1, j], ((im - ref).abs() * 8).clamp(0, 1), f"loss vs photos: {r['dG']['psnr']:.2f} dB")
fig.tight_layout(); fig.savefig(f"{OUT}/levels.png", dpi=80); plt.close(fig)

# ---- figure 2: frontier + mapping
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ks = [r["kept"] for r in sweep]; us = [r["U"]["psnr"] for r in sweep]
ax[0].plot([100 * k for k in ks], us, "o-"); [ax[0].axhline(t, color="gray", ls=":", lw=0.8) for t in (40, 35, 30)]
ax[0].set_xlabel("Gaussians kept (%)  [opacity threshold only]"); ax[0].set_ylabel("PSNR vs unpruned render (dB)"); ax[0].invert_xaxis()
ax[0].set_title("Quality-size curve for one gene (bonsai)")
ax[1].scatter([r["U"]["psnr"] for r in allr], [r["dG"]["psnr"] for r in allr], s=14)
ax[1].set_xlabel("PSNR vs unpruned render (dB)"); ax[1].set_ylabel("actual loss vs the photos (dB of PSNR)"); ax[1].set_yscale("log")
ax[1].set_title("What 'vs unpruned' means for real quality"); fig.tight_layout(); fig.savefig(f"{OUT}/frontier.png", dpi=80)
print("\ncheap fitness (half res, 8 views) vs full-res PSNR on the same views: Spearman %.3f ; vs REAL loss vs photos: %.3f" % (
    spearmanr([r["cheap"] for r in allr], [r["U"]["psnr"] for r in allr])[0], spearmanr([r["cheap"] for r in allr], [-r["dG"]["psnr"] for r in allr])[0]))
print("\nsaved figures to", OUT)
print("opacity-only sweep (kept % -> PSNR vs unpruned):", ", ".join(f"{100*r['kept']:.0f}%->{r['U']['psnr']:.0f}dB" for r in sweep[::3]))
