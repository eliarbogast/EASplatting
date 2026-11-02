"""
evaluate.py — end-to-end local check: scene + cameras -> renders -> metrics -> report.

This is the "does the whole pipeline work" script. No EA/fitness yet; it
verifies every piece the EA will rely on and writes a report you can diff
between machines (local GPU vs VACC).

Examples
  # Main benchmark: NerfBaselines 3DGS checkpoint + Mip-NeRF 360 (official images_2 for indoor)
  python evaluate.py --scene pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply \
      --cameras data/360_v2/bonsai --images images_2 --split test \
      --nb-metrics pretrained/nb/gaussian-splatting/mipnerf360/bonsai/bonsai.json \
      --out results/eval_bonsai_nb3dgs

  # Small first test: Inria pretrained truck + the dataset's raw COLMAP cameras
  python evaluate.py --scene pretrained/inria/truck/point_cloud.ply --cameras data/tandt/truck \
      --split test --out results/eval_truck_inria

  # Route B: scene you trained with gsplat + the cameras exported from its parser
  python evaluate.py --scene results/truck_gsplat_7k/ckpts/ckpt_6999_rank0.pt \
      --cameras results/truck_gsplat_7k/cameras.json --split test --lpips-net alex \
      --gsplat-stats results/truck_gsplat_7k/stats/val_step6999.json \
      --out results/eval_truck_gsplat_7k

Useful extras
  --prune-opacity 0.005   also evaluate a simply-pruned copy (count, vs-GT and vs-unpruned)
  --cache-baselines       save the unpruned renders (fp16) for the EA's fitness reference
  --fps                   synchronized single-view timing

Outputs (in --out)
  report.json   config, environment, scene/camera stats, mean metrics, timings, checks
  per_view.csv  per-view PSNR/SSIM/LPIPS (and pruned variants, if requested)
  compare/      GT | render | |error|x5 strips for the first --save-n views
  baselines_<split>.pt   (with --cache-baselines)
"""

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import torch

from cameras import _image_size, frustum_check, load_cameras, load_image
from gs_loader import activate, bytes_per_gaussian, load_scene, prune
from metrics import ImageMetrics, merge, psnr_fast, summarize
from render import measure_fps, render_views, save_png, select_indices


def environment():
    env = dict(python=platform.python_version(), torch=torch.__version__,
               cuda=torch.version.cuda, host=platform.node())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 1))
    try:
        import gsplat
        env["gsplat"] = gsplat.__version__
    except Exception as e:                       # noqa: BLE001
        env["gsplat"] = f"import failed: {e}"
    return env


def _reference_value(d, key):
    """Find `key` in a metrics json (top level or under 'metrics'); average if it's a per-image list."""
    for src in (d, d.get("metrics", {}) if isinstance(d, dict) else {}):
        if isinstance(src, dict) and key in src:
            v = src[key]
            if isinstance(v, (list, tuple)):
                return sum(v) / len(v) if v else None
            if isinstance(v, (int, float)):
                return float(v)
    return None


def crosscheck(report, path, label, lpips_net, use_lpips, hint, psnr_tol=0.05, ssim_tol=0.002, lpips_tol=0.01):
    """Compare our mean metrics with a reference json (gsplat stats/val_*.json or NerfBaselines <scene>.json)."""
    ref = json.loads(Path(path).read_text())
    mine = report["metrics_vs_gt"]
    pairs = {"psnr": "psnr", "ssim": "ssim"}
    # "vgg-raw" (Inria convention) is not comparable with either reference, so it is not checked.
    if use_lpips and lpips_net != "vgg-raw":
        # NerfBaselines: "lpips" = AlexNet, "lpips_vgg" = VGG. gsplat stats: "lpips" = whatever it ran (alex default).
        pairs["lpips"] = "lpips_vgg" if (lpips_net == "vgg" and _reference_value(ref, "lpips_vgg") is not None) else "lpips"
    refv = {k: _reference_value(ref, rk) for k, rk in pairs.items()}
    diff = {k: round(mine[k] - v, 4) for k, v in refv.items() if v is not None and k in mine}
    report.setdefault("crosschecks", {})[label] = dict(path=str(path), reference=refv, diff_ours_minus_ref=diff)
    print(f"vs {label} ({Path(path).name}): reference {refv}  diff (ours - ref) {diff}")
    if not diff:
        print(f"!! Couldn't find psnr/ssim in {path}; inspect the file's keys.")
        return
    ok = (abs(diff.get("psnr", 0)) < psnr_tol and abs(diff.get("ssim", 0)) < ssim_tol
          and abs(diff.get("lpips", 0)) < lpips_tol)
    report["checks"][f"matches_{label}"] = ok
    if not ok:
        print(f"!! Differs from {label}. {hint}")


def compare_strip(gt, pred, gain=5.0):
    """GT | render | abs-error (x gain, grayscale) side by side, (H, 3W, 3)."""
    err = ((pred - gt).abs().mean(dim=-1, keepdim=True) * gain).clamp(0, 1).expand_as(gt)
    return torch.cat([gt, pred, err], dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="point_cloud.ply or gsplat ckpt .pt")
    ap.add_argument("--cameras", required=True,
                    help="COLMAP scene dir, Inria/exported cameras.json, or transforms json/dir")
    ap.add_argument("--images", default="images", help="COLMAP image folder (images, images_2, images_4)")
    ap.add_argument("--image-dir", default=None, help="GT folder for plain Inria cameras.json")
    ap.add_argument("--blender-split", default="test")
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--k", type=int, default=None, help="k evenly spaced views from the split")
    ap.add_argument("--background", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--rasterize-mode", default="classic", choices=["classic", "antialiased"])
    ap.add_argument("--lpips-net", default="vgg", choices=["vgg", "vgg-raw", "alex"])
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-n", type=int, default=4, help="comparison strips to save")
    ap.add_argument("--cache-baselines", action="store_true")
    ap.add_argument("--prune-opacity", type=float, default=None)
    ap.add_argument("--fps", action="store_true")
    ap.add_argument("--gsplat-stats", default=None, help="gsplat stats/val_step*.json to cross-check")
    ap.add_argument("--nb-metrics", default=None,
                    help="NerfBaselines <scene>.json (from get_data.sh nb-checkpoints) to cross-check")
    ap.add_argument("--min-frustum", type=float, default=0.05)
    ap.add_argument("--force", action="store_true", help="continue despite failed checks")
    ap.add_argument("--device", default="cuda", help=argparse.SUPPRESS)   # "cpu" only for mocked tests
    args = ap.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available — gsplat rendering needs an NVIDIA GPU.")
    if dev != "cuda":                            # test harness: make CUDA timing calls no-ops
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
        torch.cuda.max_memory_allocated = lambda *a, **k: 0
    out = Path(args.out)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    report = dict(args=vars(args), env=environment(), checks={})
    print("Environment:", report["env"])

    # ---------------------------------------------------------------- scene
    t0 = time.perf_counter()
    raw = load_scene(args.scene, device=dev)
    full = activate(raw)
    N = int(raw["means"].shape[0])
    report["scene"] = dict(
        path=args.scene, num_gaussians=N, sh_degree=full["sh_degree"],
        size_mb_uncompressed=round(N * bytes_per_gaussian(raw) / 1e6, 1),
        file_mb=round(Path(args.scene).stat().st_size / 1e6, 1),
        load_s=round(time.perf_counter() - t0, 2),
        opacity_lt_0005=float((full["opacities"] < 0.005).float().mean()),
    )
    print(f"Scene: {N:,} Gaussians, SH {full['sh_degree']}, "
          f"{report['scene']['size_mb_uncompressed']} MB uncompressed")

    # -------------------------------------------------------------- cameras
    cams = load_cameras(args.cameras, images=args.images, split=args.blender_split,
                        image_dir=args.image_dir)
    idx = select_indices(len(cams), args.split, args.every, args.k)
    sub = cams.subset(idx)
    W, H = sub.size
    have_gt = bool(sub.image_paths) and all(p is not None for p in sub.image_paths)
    report["cameras"] = dict(source=args.cameras, total=len(cams), split=args.split,
                             n_views=len(idx), size=[W, H], have_gt=have_gt)
    print(f"Cameras: {len(cams)} total, {len(idx)} {args.split} views at {W}x{H}, GT images: {have_gt}")

    frac = frustum_check(raw["means"], sub)
    med = float(frac.median())
    report["checks"]["frustum_median_visible"] = med
    print(f"Frustum check: median {med:.1%} of Gaussian centers visible per view")
    if med < args.min_frustum:
        msg = ("Almost no Gaussians are in view: cameras and scene are in different frames. "
               "gsplat-trained scene? use export_gsplat_cameras.py. Blender data? check the split/background.")
        report["checks"]["frustum"] = "FAIL"
        print("!! " + msg)
        if not args.force:
            (out / "report.json").write_text(json.dumps(report, indent=1))
            sys.exit(1)

    if have_gt:
        s = _image_size(sub.image_paths[0])
        report["checks"]["gt_size_matches_camera"] = (tuple(s) == (W, H))
        if tuple(s) != (W, H):
            print(f"!! GT image is {s[0]}x{s[1]} but camera expects {W}x{H}; GT will be resized. "
                  f"Check --images / data_factor.")

    # ------------------------------------------------- render + metrics loop
    metrics = ImageMetrics(device=dev, lpips_net=args.lpips_net)
    use_lpips = not args.no_lpips and have_gt
    keep_base = args.cache_baselines or args.prune_opacity is not None
    per_view, base_chunks, t_render = {}, [], 0.0
    torch.cuda.reset_peak_memory_stats()

    for s in range(0, len(idx), args.batch_size):
        ii = list(range(s, min(s + args.batch_size, len(idx))))
        torch.cuda.synchronize(); t1 = time.perf_counter()
        imgs = render_views(full, sub, ii, batch_size=args.batch_size,
                            background=tuple(args.background), rasterize_mode=args.rasterize_mode)
        torch.cuda.synchronize(); t_render += time.perf_counter() - t1
        if keep_base:
            base_chunks.append(imgs.half().cpu())
        if have_gt:
            gt = torch.stack([load_image(sub.image_paths[j], args.background, (W, H)) for j in ii]).to(dev)
            merge(per_view, metrics(imgs, gt, lpips=use_lpips))
            for j, jj in enumerate(ii):
                if jj < args.save_n:
                    save_png(compare_strip(gt[j], imgs[j]), out / "compare" / f"{jj:03d}_{Path(str(sub.names[jj])).stem}.png")
        else:
            for j, jj in enumerate(ii):
                if jj < args.save_n:
                    save_png(imgs[j], out / "compare" / f"{jj:03d}_{Path(str(sub.names[jj])).stem}.png")

    report["timing"] = dict(render_s_per_view=round(t_render / len(idx), 4),
                            peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    if have_gt:
        report["metrics_vs_gt"] = summarize(per_view)
        m = report["metrics_vs_gt"]
        print(f"Unpruned vs GT ({args.split}, {len(idx)} views): PSNR {m['psnr']:.3f}  "
              f"SSIM {m['ssim']:.4f}" + (f"  LPIPS-{args.lpips_net} {m['lpips']:.4f}" if use_lpips else ""))
        if m["psnr"] < 15:
            report["checks"]["psnr_sane"] = False
            print("!! PSNR < 15 dB: likely camera, image-folder, or background mismatch. Look at compare/.")
        else:
            report["checks"]["psnr_sane"] = True

    base = torch.cat(base_chunks) if keep_base else None
    if args.cache_baselines:
        p = out / f"baselines_{args.split}.pt"
        torch.save(dict(images=base, indices=idx, names=list(sub.names), scene_path=args.scene,
                        num_gaussians=N, size=(W, H),
                        render_kw=dict(background=tuple(args.background), rasterize_mode=args.rasterize_mode)), p)
        report["baselines"] = str(p)
        print(f"Cached {len(idx)} baseline renders -> {p} ({p.stat().st_size / 1e6:.0f} MB)")

    # --------------------------------------------- optional: simple pruning
    if args.prune_opacity is not None:
        mask = full["opacities"] > args.prune_opacity
        pruned = prune(full, mask)
        kept = int(mask.sum())
        pv, vs_base = {}, []
        for s in range(0, len(idx), args.batch_size):
            ii = list(range(s, min(s + args.batch_size, len(idx))))
            imgs = render_views(pruned, sub, ii, batch_size=args.batch_size,
                                background=tuple(args.background), rasterize_mode=args.rasterize_mode)
            vs_base += psnr_fast(imgs, base[s:s + len(ii)].to(dev).float(), max_db=60).tolist()
            if have_gt:
                gt = torch.stack([load_image(sub.image_paths[j], args.background, (W, H)) for j in ii]).to(dev)
                merge(pv, metrics(imgs, gt, lpips=use_lpips))
        res = dict(threshold=args.prune_opacity, kept=kept, kept_frac=kept / N,
                   psnr_vs_unpruned_capped60=sum(vs_base) / len(vs_base))
        if have_gt:
            res["metrics_vs_gt"] = summarize(pv)
            for k, v in pv.items():
                per_view["pruned_" + k] = v
        report["prune_demo"] = res
        print(f"Prune opacity > {args.prune_opacity}: kept {kept:,}/{N:,} ({kept / N:.1%}); "
              f"PSNR vs unpruned {res['psnr_vs_unpruned_capped60']:.2f} dB"
              + (f"; vs GT {res['metrics_vs_gt']['psnr']:.3f}" if have_gt else ""))

    # ------------------------------------------------------------- timing
    if args.fps:
        r = measure_fps(full, sub, background=tuple(args.background), rasterize_mode=args.rasterize_mode)
        report["fps_unpruned"] = r
        print(f"FPS (unpruned, single view): {r['fps']:.1f} ({r['ms_per_frame']:.2f} ms)")
        if args.prune_opacity is not None:
            r2 = measure_fps(pruned, sub, background=tuple(args.background), rasterize_mode=args.rasterize_mode)
            report["fps_pruned"] = r2
            print(f"FPS (pruned):                {r2['fps']:.1f} ({r2['ms_per_frame']:.2f} ms)")

    # ------------------------------- cross-check vs published / trainer eval
    if have_gt:
        if args.gsplat_stats:
            crosscheck(report, args.gsplat_stats, "gsplat", args.lpips_net, use_lpips,
                       hint="Same --split/views? Same LPIPS net (gsplat default: alex)?")
        if args.nb_metrics:
            crosscheck(report, args.nb_metrics, "nerfbaselines", args.lpips_net, use_lpips,
                       hint="NB protocol: --split test, official images_4 (outdoor) / images_2 (indoor), "
                            "--lpips-net vgg (not vgg-raw). Small diffs (<0.1 dB) can come from rasterizer differences.",
                       psnr_tol=0.1, ssim_tol=0.003)

    # -------------------------------------------------------------- write
    if per_view:
        with open(out / "per_view.csv", "w", newline="") as f:
            w = csv.writer(f)
            keys = list(per_view)
            w.writerow(["index", "name"] + keys)
            for j in range(len(idx)):
                w.writerow([idx[j], sub.names[j]] + [round(per_view[k][j], 6) for k in keys])
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"Report -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
