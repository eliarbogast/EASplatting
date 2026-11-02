"""
render.py — render 3DGS scenes with gsplat, cache baseline renders, sanity checks.

Core pieces for the EA loop
  render_views(splats, cams, idx)      -> (V, H, W, 3) images in [0, 1]
  cache_baselines(...) / load_baselines(...)
                                       render the UNPRUNED scene once, save to disk
  gaussian_contributions(splats, cams) -> per-Gaussian visibility + total blend
                                       weight (an importance score for pruning)
  measure_fps(splats, cams)            -> properly synchronized render timing

`splats` is an ACTIVATED dict from gs_loader.activate(), so the EA can activate
the full scene once and then just mask it:

    raw    = load_scene("point_cloud.ply", device="cuda")
    full   = activate(raw)
    cams   = load_cameras("data/garden", images="images_4").to("cuda")
    base   = load_baselines("cache/garden_train.pt", device="cuda")
    imgs   = render_views(prune(full, mask), cams, base["indices"])
    loss   = psnr(imgs, base["images"])

Command line (needs a CUDA GPU + `pip install gsplat`):
    python render.py --scene point_cloud.ply --cameras data/garden --images images_4 \
        --split train --every 8 --out cache/garden_train.pt --png-dir renders/ \
        --check-gt --demo-prune 0.005 --fps

Requires: torch, numpy, gsplat, Pillow (and gs_loader.py, cameras.py alongside).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cameras import Cameras, evenly_spaced, load_cameras, load_image, split_indices
from gs_loader import activate, load_scene, prune

# ------------------------------------------------------------------- rendering


def _rasterize(splats, viewmats, Ks, W, H, *, colors=None, sh_degree="auto",
               backgrounds=None, near=0.01, far=1e10, rasterize_mode="classic", packed=True):
    """Thin wrapper over gsplat.rasterization (imported lazily so CPU-only tools work)."""
    from gsplat import rasterization
    if colors is None:
        colors = splats["colors"]
    if sh_degree == "auto":
        sh_degree = splats["sh_degree"]
    return rasterization(
        means=splats["means"], quats=splats["quats"], scales=splats["scales"],
        opacities=splats["opacities"], colors=colors,
        viewmats=viewmats, Ks=Ks, width=W, height=H,
        near_plane=near, far_plane=far, sh_degree=sh_degree,
        backgrounds=backgrounds, packed=packed, rasterize_mode=rasterize_mode,
    )


@torch.no_grad()
def render_views(splats, cams: Cameras, indices=None, *, batch_size=8,
                 background=(0.0, 0.0, 0.0), rasterize_mode="classic",
                 near=0.01, far=1e10, out_device=None, dtype=None):
    """
    Render cameras `indices` (default: all) of `splats` (activated dict).
    Returns (V, H, W, 3) clamped to [0, 1]. All selected cameras must share
    one image size (true for standard benchmarks).

    rasterize_mode: "classic" for Inria / default-trained scenes; use
    "antialiased" only if the scene was trained with antialiasing.
    background: must be identical for baseline and pruned renders
    (white for NeRF-Synthetic scenes trained with --white_background).
    """
    indices = list(range(len(cams))) if indices is None else list(indices)
    sub = cams.subset(indices)
    W, H = sub.size
    device = splats["means"].device
    vm, Ks = sub.viewmats.to(device), sub.Ks.to(device)
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    out_device = device if out_device is None else out_device

    # Empty scene (an over-aggressive genome): just the background, no rasterizer call.
    if splats["means"].shape[0] == 0:
        img = bg.expand(len(indices), H, W, 3).clone()
        return img.to(out_device, dtype or img.dtype)

    chunks = []
    for s in range(0, len(indices), batch_size):
        e = min(s + batch_size, len(indices))
        # Composite the background ourselves: gsplat 1.5.3 asserts on per-camera `backgrounds`
        # when packed=True (it checks against the packed shape). Rendering on black and adding
        # (1 - alpha) * bg is exactly what the rasterizer does internally.
        colors, alphas, _ = _rasterize(splats, vm[s:e], Ks[s:e], W, H,
                                       near=near, far=far, rasterize_mode=rasterize_mode)
        img = colors[..., :3]
        if bool((bg != 0).any()):
            img = img + (1 - alphas) * bg
        img = img.clamp_(0, 1)
        chunks.append(img.to(out_device, dtype or img.dtype))
    return torch.cat(chunks, dim=0)


# ------------------------------------------------------------ baseline caching


def cache_baselines(scene_path, cams: Cameras, indices, out_path, *,
                    device="cuda", dtype=torch.float16, **render_kw):
    """
    Render the unpruned scene from `indices` once and save to `out_path`.
    float16 keeps files ~half size; its rounding error (~1e-3) is far below
    any pruning damage you'll measure (PSNR ceiling ~ 65+ dB).
    """
    raw = load_scene(scene_path, device=device)
    full = activate(raw)
    imgs = render_views(full, cams, indices, out_device="cpu", dtype=dtype, **render_kw)
    payload = dict(
        images=imgs, indices=list(indices), names=[cams.names[i] for i in indices],
        scene_path=str(scene_path), num_gaussians=int(raw["means"].shape[0]),
        size=cams.subset(indices).size, render_kw=render_kw,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload


def load_baselines(path, device="cuda", dtype=torch.float32):
    """Load cached baselines; images moved to `device` as `dtype` for metric math."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    d["images"] = d["images"].to(device, dtype)
    return d


# -------------------------------------------------------- per-Gaussian scores


def gaussian_contributions(splats, cams: Cameras, indices=None, *, batch_size=4,
                           near=0.01, far=1e10, rasterize_mode="classic"):
    """
    Per-Gaussian statistics over the given views, for importance-based genes:

      visible_views (N,)  number of views where the Gaussian has nonzero
                          screen-space radius (i.e. survives culling)
      blend_weight  (N,)  sum over all pixels and views of its alpha-blending
                          weight  w_i(p) = T_i(p) * alpha_i(p)

    blend_weight is computed exactly with one backward pass: render a constant
    1-channel "color" per Gaussian; since the image is linear in colors,
    d(sum of image)/d(color_i) = sum_p w_i(p). It already folds in opacity and
    occlusion, so it's a cleaner version of LightGaussian's
    hit-count x opacity x volume score. Compute once per scene and cache.
    """
    indices = list(range(len(cams))) if indices is None else list(indices)
    sub = cams.subset(indices)
    W, H = sub.size
    device = splats["means"].device
    N = splats["means"].shape[0]
    vm, Ks = sub.viewmats.to(device), sub.Ks.to(device)

    ones = torch.ones(N, 1, device=device, requires_grad=True)
    visible = torch.zeros(N, dtype=torch.long, device=device)
    detached = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in splats.items()}

    for s in range(0, len(indices), batch_size):
        e = min(s + batch_size, len(indices))
        img, _, meta = _rasterize(detached, vm[s:e], Ks[s:e], W, H, colors=ones,
                                  sh_degree=None, near=near, far=far,
                                  rasterize_mode=rasterize_mode, packed=False)
        img.sum().backward()
        radii = meta["radii"]                    # (C, N, 2) in gsplat >= 1.5; (C, N) in older versions
        vis = (radii > 0).all(dim=-1) if radii.ndim == 3 else (radii > 0)   # same rule gsplat uses
        visible += vis.sum(dim=0)

    return dict(visible_views=visible, blend_weight=ones.grad[:, 0].detach())


# ------------------------------------------------------------------ timing


@torch.no_grad()
def measure_fps(splats, cams: Cameras, indices=None, *, n_warmup=10, n_iters=100,
                background=(0.0, 0.0, 0.0), rasterize_mode="classic"):
    """
    Single-view render timing (what "FPS" means in 3DGS papers).
    Synchronizes the GPU around every render; reports the median.
    Compare numbers only across runs on the same GPU type and resolution.
    """
    indices = list(range(len(cams))) if indices is None else list(indices)
    times = []
    for it in range(n_warmup + n_iters):
        i = indices[it % len(indices)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        render_views(splats, cams, [i], batch_size=1, background=background,
                     rasterize_mode=rasterize_mode)
        torch.cuda.synchronize()
        if it >= n_warmup:
            times.append(time.perf_counter() - t0)
    ms = 1000 * float(np.median(times))
    return dict(ms_per_frame=ms, fps=1000 / ms)


# ----------------------------------------------------------------- helpers


def psnr(a, b, max_db=None):
    """
    Mean per-view PSNR (dB) between (V, H, W, 3) image stacks in [0, 1].
    Identical images give inf; pass max_db (e.g. 60) to cap for fitness use.
    Placeholder until metrics.py exists — use one implementation everywhere.
    """
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(dim=1)
    db = -10 * torch.log10(mse)
    if max_db is not None:
        db = db.clamp(max=max_db)
    return db.mean().item()


def save_png(img, path):
    from PIL import Image
    arr = (img.detach().float().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def select_indices(n, split="train", every=1, k=None):
    """Indices for a split ("train"/"test"/"all"), optionally every n-th or k evenly spaced."""
    train, test = split_indices(n)
    idx = {"train": train, "test": test, "all": list(range(n))}[split]
    idx = idx[::every]
    return evenly_spaced(idx, k) if k else idx


# --------------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="point_cloud.ply or gsplat .pt")
    ap.add_argument("--cameras", required=True, help="COLMAP dir, cameras.json, or transforms json/dir")
    ap.add_argument("--images", default="images", help="COLMAP image folder, e.g. images_4")
    ap.add_argument("--image-dir", default=None, help="GT images for cameras.json")
    ap.add_argument("--blender-split", default="train", help="split file for NeRF-Synthetic")
    ap.add_argument("--split", default="train", choices=["train", "test", "all"],
                    help="every-8th split (COLMAP / cameras.json); use 'all' for Blender files")
    ap.add_argument("--every", type=int, default=1, help="take every n-th camera of the split")
    ap.add_argument("--k", type=int, default=None, help="or: k evenly spaced cameras")
    ap.add_argument("--background", type=float, nargs=3, default=[0, 0, 0])
    ap.add_argument("--rasterize-mode", default="classic", choices=["classic", "antialiased"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None, help="save baseline renders here (.pt)")
    ap.add_argument("--png-dir", default=None, help="write the first few renders as PNGs")
    ap.add_argument("--check-gt", action="store_true", help="PSNR of unpruned renders vs GT photos")
    ap.add_argument("--demo-prune", type=float, default=None,
                    help="prune at this min opacity and report count + PSNR vs baseline")
    ap.add_argument("--fps", action="store_true", help="measure render FPS")
    args = ap.parse_args()

    device = "cuda"
    cams = load_cameras(args.cameras, images=args.images, split=args.blender_split,
                        image_dir=args.image_dir)
    idx = select_indices(len(cams), args.split, args.every, args.k)
    rkw = dict(background=tuple(args.background), rasterize_mode=args.rasterize_mode)
    print(f"{cams}; rendering {len(idx)} views ({args.split})")

    raw = load_scene(args.scene, device=device)
    full = activate(raw)
    N = raw["means"].shape[0]
    print(f"Scene: {N:,} Gaussians, SH degree {full['sh_degree']}")

    t0 = time.perf_counter()
    base = render_views(full, cams, idx, batch_size=args.batch_size, **rkw)
    torch.cuda.synchronize()
    print(f"Rendered {len(idx)} views at {cams.subset(idx).size} in {time.perf_counter() - t0:.2f}s")

    if args.out:
        torch.save(dict(images=base.half().cpu(), indices=idx, names=[cams.names[i] for i in idx],
                        scene_path=args.scene, num_gaussians=N, size=cams.subset(idx).size,
                        render_kw=rkw), args.out)
        print(f"Saved baselines -> {args.out}")

    if args.png_dir:
        for j, i in enumerate(idx[:4]):
            save_png(base[j], Path(args.png_dir) / f"{Path(str(cams.names[i])).stem}.png")
        print(f"Wrote {min(4, len(idx))} PNGs -> {args.png_dir}")

    if args.check_gt:
        have = [(j, i) for j, i in enumerate(idx) if cams.image_paths and cams.image_paths[i]]
        if not have:
            print("No GT images available for these cameras.")
        else:
            W, H = cams.subset(idx).size
            gt = torch.stack([load_image(cams.image_paths[i], args.background, (W, H)) for _, i in have])
            p = psnr(base[[j for j, _ in have]].cpu(), gt)
            print(f"Unpruned vs GT photos: {p:.2f} dB over {len(have)} views "
                  f"(expect mid-20s to 30s on train views; < 15 dB suggests a camera/background bug)")

    if args.demo_prune is not None:
        mask = full["opacities"] > args.demo_prune
        pruned = prune(full, mask)
        imgs = render_views(pruned, cams, idx, batch_size=args.batch_size, **rkw)
        kept = int(mask.sum())
        print(f"Demo prune (opacity > {args.demo_prune}): kept {kept:,}/{N:,} ({kept / N:.1%}), "
              f"PSNR vs unpruned {psnr(imgs, base, max_db=60):.2f} dB (capped at 60)")

    if args.fps:
        r = measure_fps(full, cams, idx, **rkw)
        print(f"FPS (unpruned, 1 view/frame): {r['fps']:.1f}  ({r['ms_per_frame']:.2f} ms) on "
              f"{torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
