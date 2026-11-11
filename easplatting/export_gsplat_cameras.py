"""
export_gsplat_cameras.py — write out the exact cameras gsplat's simple_trainer trained with.

Why this exists
  simple_trainer normalizes world space by default (normalize_world_space=True):
  it applies a similarity transform (recenter + rescale + align principal axes,
  sometimes a 180-degree flip) to the cameras AND the points. The trained splats
  therefore live in that normalized frame, and the raw COLMAP poses in
  sparse/0 do NOT line up with them (the frustum check in cameras.py would show
  ~0% visible). Inria-trained scenes don't normalize, so they pair with raw COLMAP.

  This script runs gsplat's own Parser with the same settings the trainer used
  and writes an Inria-style cameras.json (+ cx, cy, image_path per entry) that
  cameras.load_cameras() reads directly. image_path points at the exact images
  the trainer evaluated against (for data_factor > 1 with JPEGs, gsplat makes
  its own images_<f>_png folder by downscaling the full-res images).

Run from the project root, in the env that runs simple_trainer:
    python -m easplatting.export_gsplat_cameras --data data/tandt/truck --factor 1 \
        --examples ext/gsplat/examples --out results/truck_gsplat_7k/cameras.json

--factor / --test-every / --normalize must match the training run's
--data_factor / --test_every / --normalize_world_space.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="COLMAP scene dir used for training")
    ap.add_argument("--factor", type=int, default=4, help="the trainer's --data_factor (default 4)")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                    help="the trainer's normalize_world_space (default True)")
    ap.add_argument("--examples", default="ext/gsplat/examples", help="path to gsplat/examples")
    ap.add_argument("--out", required=True, help="output cameras.json")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data)
    sys.path.insert(0, os.path.abspath(args.examples))
    cwd = os.getcwd()
    os.chdir(args.examples)                      # gsplat's examples import siblings (utils, exif, ...)
    try:
        from datasets.colmap import Parser
        p = Parser(data_dir=data_dir, factor=args.factor, normalize=args.normalize,
                   test_every=args.test_every)
    finally:
        os.chdir(cwd)

    entries, distorted = [], False
    for i, name in enumerate(p.image_names):
        cid = p.camera_ids[i]
        K = np.asarray(p.Ks_dict[cid], dtype=np.float64)
        W, H = p.imsize_dict[cid]
        if len(p.params_dict[cid]) > 0:
            distorted = True
        c2w = np.asarray(p.camtoworlds[i], dtype=np.float64)
        entries.append(dict(
            id=i,
            img_name=os.path.splitext(name)[0],
            width=int(W), height=int(H),
            position=c2w[:3, 3].tolist(),
            rotation=c2w[:3, :3].tolist(),
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            image_path=os.path.abspath(p.image_paths[i]),
        ))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=1))
    meta = dict(
        source="gsplat datasets.colmap.Parser",
        data_dir=data_dir, factor=args.factor, test_every=args.test_every,
        normalize_world_space=args.normalize,
        world_transform=np.asarray(p.transform).tolist(),   # normalized = T @ raw COLMAP world
        scene_scale=float(getattr(p, "scene_scale", float("nan"))),
        num_images=len(entries),
    )
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=1))

    print(f"Wrote {len(entries)} cameras -> {out}  ({entries[0]['width']}x{entries[0]['height']}, "
          f"normalize={args.normalize}, factor={args.factor})")
    if distorted:
        print("WARNING: non-PINHOLE COLMAP cameras. gsplat undistorts images on the fly; image_path "
              "points at the DISTORTED originals, so GT metrics will be off. Use undistorted data "
              "(Mip-NeRF 360, Inria's tandt_db are already undistorted).")


if __name__ == "__main__":
    main()
