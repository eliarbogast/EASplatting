"""
bench_eval.py — what does one EA fitness evaluation cost on THIS GPU? Re-run it on every new machine
(laptop, each VACC GPU type) and update the compute budget in docs/EA_PLAN.md from the numbers.

    python scripts/bench_eval.py            # from the repo root, env active, GPU visible (~2 min)

An evaluation = threshold mask -> prune -> render 8 train views at half resolution -> PSNR vs cached
baselines. Reports (A) cost vs scene size, incl. synthetic 5M/10M-Gaussian scenes made by tiling bonsai
(a rough proxy for the outdoor Mip-NeRF 360 scenes; needs the bonsai data from get_data.sh, and the 7k-step
truck from train_gsplat.sh if you want that row), (B) whether one process already saturates the GPU
(aggregate evals/s with 1, 2, 4 processes), (C) the extra cost of SSIM / LPIPS inside the loop.
"""
import sys, time, statistics as st, subprocess, os
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cameras import Cameras, load_cameras, split_indices, evenly_spaced
from gs_loader import load_scene, activate, prune
from render import render_views
from metrics import psnr_fast, ImageMetrics
from viewer import gene_stats, keep_mask
dev = "cuda"
BONSAI = ("pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply", "data/360_v2/bonsai", "images_2")
TRUCK = ("results/truck_gsplat_7k/ckpts/ckpt_6999_rank0.pt", "results/truck_gsplat_7k/cameras.json", None)

def setup(spec, copies=1):
    act = activate(load_scene(spec[0], device=dev))
    if copies > 1:   # synthetic bigger scene: jittered copies (overdraw-heavy, a rough proxy for denser scenes)
        g = torch.Generator(device=dev).manual_seed(0); parts = []
        for c in range(copies):
            p = dict(act); p["means"] = act["means"] + 0.01 * torch.randn(act["means"].shape, device=dev, generator=g); parts.append(p)
        act = {k: (torch.cat([p[k] for p in parts]) if torch.is_tensor(act[k]) and act[k].ndim > 0 else act[k]) for k in act}
    cams = load_cameras(spec[1], images=spec[2]) if spec[2] else load_cameras(spec[1])
    train, _ = split_indices(len(cams), 8)
    c = cams.subset(evenly_spaced(train, 8)); Ks = c.Ks.clone(); Ks[:, :2, :] *= 0.5
    fc = Cameras(names=c.names, viewmats=c.viewmats, Ks=Ks, widths=[int(w / 2) for w in c.widths], heights=[int(h / 2) for h in c.heights])
    return act, fc

def timed(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

def worker():
    act, fc = setup(BONSAI); stats = gene_stats(act); base = render_views(act, fc, batch_size=8)
    th = (0.01, 10 ** 0.4, 10 ** 3.0)
    def ev():
        sp = prune(act, keep_mask(stats, *th)); return float(psnr_fast(render_views(sp, fc, batch_size=8), base).mean())
    ev(); torch.cuda.synchronize(); t = time.perf_counter(); n = 150
    for _ in range(n): ev()
    torch.cuda.synchronize(); print(f"{n / (time.perf_counter() - t):.1f}")

if len(sys.argv) > 1 and sys.argv[1] == "worker": worker(); sys.exit()

print("== A. cost vs scene size (K=8 views @ half res, unpruned scene; ms)")
print(f"{'scene':<28} {'N':>11} {'render':>8} {'prune-copy':>10} {'zero-opacity eval':>18} {'peak GB':>8}")
rows = [("bonsai (NB 3DGS)", BONSAI, 1), ("truck (gsplat 7k)", TRUCK, 1), ("bonsai x4 (synthetic)", BONSAI, 4), ("bonsai x8 (synthetic)", BONSAI, 8)]
for label, spec, copies in [r for r in rows if Path(r[1][0]).exists() or print(f"(skipping {r[0]}: {r[1][0]} not found)")]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    act, fc = setup(spec, copies); N = act["means"].shape[0]; stats = gene_stats(act)
    mask = keep_mask(stats, 0.0, float("inf"), float("inf"))     # keep everything = worst case for render time
    r = timed(lambda: render_views(act, fc, batch_size=8))
    p = timed(lambda: prune(act, mask))
    def zero():
        sp = dict(act); sp["opacities"] = act["opacities"] * mask; return render_views(sp, fc, batch_size=8)
    z = timed(zero)
    print(f"{label:<28} {N:>11,} {r:8.1f} {p:10.1f} {z:18.1f} {torch.cuda.max_memory_allocated() / 1e9:8.2f}")
    del act, stats, mask; torch.cuda.empty_cache()

print("\n== B. does the GPU saturate? aggregate evals/s with P identical processes (bonsai, K=8 @ half res, realistic genome)")
for P in (1, 2, 4):
    procs = [subprocess.Popen([sys.executable, __file__, "worker"], stdout=subprocess.PIPE, text=True,
                              env=dict(os.environ)) for _ in range(P)]
    rates = [float(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
    print(f"  {P} process(es): per-process {st.mean(rates):5.1f} evals/s, aggregate {sum(rates):5.1f} evals/s")

print("\n== C. cost of richer metrics inside the loop (8 views @ half res, 779x519; ms per evaluation)")
act, fc = setup(BONSAI); base = render_views(act, fc, batch_size=8)
sp = prune(act, keep_mask(gene_stats(act), 0.01, 10 ** 0.4, 10 ** 3.0)); imgs = render_views(sp, fc, batch_size=8)
m = ImageMetrics(device=dev, lpips_net="vgg"); ma = ImageMetrics(device=dev, lpips_net="alex")
print(f"  render only                : {timed(lambda: render_views(sp, fc, batch_size=8)):6.1f}")
print(f"  + PSNR (psnr_fast)         : {timed(lambda: psnr_fast(imgs, base)):6.2f}   (extra)")
print(f"  + SSIM+PSNR (torchmetrics) : {timed(lambda: m(imgs, base, lpips=False), 5):6.1f}   (extra)")
print(f"  + LPIPS-alex               : {timed(lambda: ma(imgs, base, lpips=True), 3) - timed(lambda: ma(imgs, base, lpips=False), 3):6.1f}   (extra)")
print(f"  + LPIPS-vgg                : {timed(lambda: m(imgs, base, lpips=True), 3) - timed(lambda: m(imgs, base, lpips=False), 3):6.1f}   (extra)")
