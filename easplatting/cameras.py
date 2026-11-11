"""
cameras.py — load camera poses + intrinsics for 3DGS scenes into gsplat's format.

Supported sources
  * COLMAP sparse model        <data>/sparse/0/{cameras,images}.{bin,txt}
                               (Mip-NeRF 360, Tanks & Temples, Deep Blending)
  * Inria cameras.json         shipped next to point_cloud.ply in the Inria
                               pretrained models — enough to render baselines
                               WITHOUT downloading the dataset images
  * Blender / NeRF-Synthetic   transforms_{train,test}.json (also basic
                               nerfstudio-style transforms.json)

Everything is converted to one representation:

  Cameras.viewmats  (C, 4, 4)  world -> camera, OpenCV convention
                               (x right, y down, z forward)   <- what gsplat wants
  Cameras.Ks        (C, 3, 3)  pinhole intrinsics, scaled to the image size used
  Cameras.widths / heights     per-camera image size (ints)
  Cameras.names / image_paths  for loading ground-truth photos

Train/test split: images are sorted by name and every 8th one is test
(`split_indices`), matching the Inria 3DGS repo (llffhold=8) and gsplat.
NeRF-Synthetic instead has separate transforms_train/test.json files.

Sanity check from the command line (CPU only, no gsplat needed):
    python -m easplatting.cameras <data_dir | cameras.json | transforms.json> [--images images_4]
                      [--scene point_cloud.ply]
`--scene` runs a frustum check: what fraction of Gaussian centers land in front
of / inside each camera. Near 0% means a pose-convention bug.
"""

import argparse
import json
import math
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------- container


@dataclass
class Cameras:
    names: list
    viewmats: torch.Tensor          # (C, 4, 4) world->camera, OpenCV
    Ks: torch.Tensor                # (C, 3, 3)
    widths: list
    heights: list
    image_paths: list = field(default_factory=list)   # may contain None

    def __len__(self):
        return len(self.names)

    def subset(self, idx):
        idx = list(idx)
        return Cameras(
            names=[self.names[i] for i in idx],
            viewmats=self.viewmats[idx],
            Ks=self.Ks[idx],
            widths=[self.widths[i] for i in idx],
            heights=[self.heights[i] for i in idx],
            image_paths=[self.image_paths[i] for i in idx] if self.image_paths else [],
        )

    def to(self, device):
        return Cameras(self.names, self.viewmats.to(device), self.Ks.to(device),
                       self.widths, self.heights, self.image_paths)

    @property
    def size(self):
        """(W, H) if all cameras share one image size (needed for batched rendering)."""
        sizes = set(zip(self.widths, self.heights))
        if len(sizes) != 1:
            raise ValueError(f"Cameras have {len(sizes)} different image sizes: {sorted(sizes)[:5]}")
        return sizes.pop()

    def centers(self):
        """Camera centers in world coords, (C, 3): c = -R^T t."""
        R, t = self.viewmats[:, :3, :3], self.viewmats[:, :3, 3:]
        return (-R.transpose(1, 2) @ t)[..., 0]

    def __repr__(self):
        sizes = sorted(set(zip(self.widths, self.heights)))
        return f"Cameras(n={len(self)}, sizes={sizes[:3]}{'...' if len(sizes) > 3 else ''})"


def _K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _pack(names, w2cs, Ks, widths, heights, paths):
    return Cameras(
        names=list(names),
        viewmats=torch.from_numpy(np.stack(w2cs)).float(),
        Ks=torch.from_numpy(np.stack(Ks)).float(),
        widths=[int(w) for w in widths],
        heights=[int(h) for h in heights],
        image_paths=list(paths),
    )


def _image_size(path):
    """(W, H) from the image header, or None if the file is missing."""
    if path is None or not Path(path).exists():
        return None
    from PIL import Image
    with Image.open(path) as im:
        return im.size


# ------------------------------------------------------------------------ COLMAP

# model_id -> (name, number of params)
COLMAP_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
_SINGLE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                 "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}


def qvec_to_rotmat(q):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def _read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            model, n_params = COLMAP_MODELS[model_id]
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cams[cam_id] = (model, w, h, params)
    return cams


def _read_images_bin(path):
    imgs = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (_image_id,) = struct.unpack("<i", f.read(4))
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            (cam_id,) = struct.unpack("<i", f.read(4))
            name = bytearray()
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n_pts, 1)                       # skip 2D points (x, y, id)
            imgs.append((name.decode(), q, t, cam_id))
    return imgs


def _read_cameras_txt(path):
    cams = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tok = line.split()
        cams[int(tok[0])] = (tok[1], int(tok[2]), int(tok[3]), tuple(map(float, tok[4:])))
    return cams


def _read_images_txt(path):
    imgs = []
    lines = [l for l in Path(path).read_text().splitlines() if not l.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():                         # stray blank line
            i += 1
            continue
        tok = lines[i].split()
        q, t = tuple(map(float, tok[1:5])), tuple(map(float, tok[5:8]))
        imgs.append((" ".join(tok[9:]), q, t, int(tok[8])))
        i += 2                                           # skip the 2D-points line (may be empty)
    return imgs


def load_colmap(data_dir, images="images", sparse=None):
    """
    Load a COLMAP scene. `images` is the image folder used for training/eval,
    e.g. "images_4" for Mip-NeRF 360 outdoor scenes; intrinsics are rescaled
    to that folder's actual image size.
    """
    data_dir = Path(data_dir)
    if sparse is None:
        sparse = next((p for p in (data_dir / "sparse" / "0", data_dir / "sparse")
                       if (p / "cameras.bin").exists() or (p / "cameras.txt").exists()), None)
        if sparse is None:
            raise FileNotFoundError(f"No COLMAP sparse model under {data_dir}/sparse[/0]")
    sparse = Path(sparse)

    if (sparse / "cameras.bin").exists():
        cams, imgs = _read_cameras_bin(sparse / "cameras.bin"), _read_images_bin(sparse / "images.bin")
    else:
        cams, imgs = _read_cameras_txt(sparse / "cameras.txt"), _read_images_txt(sparse / "images.txt")

    imgs.sort(key=lambda r: r[0])                        # sort by name -> stable split
    img_dir = data_dir / images
    # Fallback downscale factor if image files aren't present, e.g. "images_4" -> 4
    try:
        fallback = float(images.rsplit("_", 1)[1]) if "_" in images else 1.0
    except ValueError:
        fallback = 1.0

    warned, names, w2cs, Ks, Ws, Hs, paths = set(), [], [], [], [], [], []
    for name, q, t, cam_id in imgs:
        model, w, h, p = cams[cam_id]
        if model not in ("PINHOLE", "SIMPLE_PINHOLE") and model not in warned:
            warnings.warn(f"COLMAP model {model}: distortion ignored. Undistort first "
                          f"(colmap image_undistorter) for correct renders.")
            warned.add(model)
        if model in _SINGLE_FOCAL:
            fx = fy = p[0]; cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[:4]

        path = img_dir / name
        size = _image_size(path)
        if size is None:
            size = (round(w / fallback), round(h / fallback))
            path = None
        sx, sy = size[0] / w, size[1] / h

        w2c = np.eye(4)
        w2c[:3, :3] = qvec_to_rotmat(q)
        w2c[:3, 3] = t
        names.append(name); w2cs.append(w2c); paths.append(path)
        Ks.append(_K(fx * sx, fy * sy, cx * sx, cy * sy))
        Ws.append(size[0]); Hs.append(size[1])

    if all(p is None for p in paths):
        warnings.warn(f"No images found in {img_dir}; sizes inferred from COLMAP / folder name.")
    return _pack(names, w2cs, Ks, Ws, Hs, paths)


# ------------------------------------------------------------- Inria cameras.json


def load_inria_cameras_json(path, image_dir=None):
    """
    Load the cameras.json the Inria 3DGS repo writes next to point_cloud.ply.
    Each entry stores camera->world `rotation` (3x3) and `position`, OpenCV
    convention, plus fx, fy, width, height (at the training resolution).
    The principal point is not stored; Inria assumes the image center.
    Files from export_gsplat_cameras.py add cx, cy and image_path per entry.
    """
    entries = sorted(json.loads(Path(path).read_text()), key=lambda e: e["img_name"])
    names, w2cs, Ks, Ws, Hs, paths = [], [], [], [], [], []
    for e in entries:
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(e["rotation"])
        c2w[:3, 3] = np.array(e["position"])
        W, H = e["width"], e["height"]
        p = None
        if e.get("image_path"):                          # written by export_gsplat_cameras.py
            p = Path(e["image_path"])
            p = p if p.is_absolute() else Path(path).parent / p
            p = p if p.exists() else None
        elif image_dir is not None:
            p = next(iter(sorted(Path(image_dir).glob(e["img_name"] + ".*"))), None)
        names.append(e["img_name"]); w2cs.append(np.linalg.inv(c2w)); paths.append(p)
        # cx/cy are an extension (export_gsplat_cameras.py); Inria files assume the center.
        Ks.append(_K(e["fx"], e["fy"], e.get("cx", W / 2), e.get("cy", H / 2)))
        Ws.append(W); Hs.append(H)
    return _pack(names, w2cs, Ks, Ws, Hs, paths)


# ------------------------------------------------------------ Blender / NeRF-Synthetic


def load_blender(path, split="train"):
    """
    Load NeRF-Synthetic-style transforms. `path` is either the json file or the
    scene directory (then transforms_{split}.json, else transforms.json).
    transform_matrix is camera->world in OpenGL convention (y up, -z forward);
    it is flipped to OpenCV and inverted to world->camera.
    """
    path = Path(path)
    if path.is_dir():
        cand = path / f"transforms_{split}.json"
        path = cand if cand.exists() else path / "transforms.json"
    meta = json.loads(path.read_text())
    root = path.parent

    names, w2cs, Ks, Ws, Hs, paths = [], [], [], [], [], []
    for fr in meta["frames"]:
        g = lambda k, d=None: fr.get(k, meta.get(k, d))   # per-frame overrides global
        fp = root / fr["file_path"]
        img = next((c for c in (fp, fp.with_name(fp.name + ".png"), fp.with_name(fp.name + ".jpg"))
                    if c.is_file()), None)
        size = (g("w"), g("h")) if g("w") and g("h") else _image_size(img)
        if size is None:
            size = (800, 800)                            # NeRF-Synthetic default
            warnings.warn(f"Image size unknown for {fp}; assuming 800x800.")
        W, H = int(size[0]), int(size[1])

        if g("fl_x") is not None:
            fx = g("fl_x"); fy = g("fl_y", fx)
        else:
            fx = fy = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
        cx, cy = g("cx", W / 2), g("cy", H / 2)

        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1                               # OpenGL -> OpenCV (flip y, z axes)
        names.append(fr["file_path"]); w2cs.append(np.linalg.inv(c2w)); paths.append(img)
        Ks.append(_K(fx, fy, cx, cy)); Ws.append(W); Hs.append(H)
    return _pack(names, w2cs, Ks, Ws, Hs, paths)


# ---------------------------------------------------------------------- dispatch


def load_cameras(path, images="images", split="train", image_dir=None):
    """Pick the right loader from what `path` points at."""
    path = Path(path)
    if path.is_file() and path.name == "cameras.json":
        return load_inria_cameras_json(path, image_dir=image_dir)
    if path.is_file() and path.suffix == ".json":
        return load_blender(path, split=split)
    if path.is_dir():
        if (path / "sparse").exists():
            return load_colmap(path, images=images)
        if any(path.glob("transforms*.json")):
            return load_blender(path, split=split)
        if (path / "cameras.json").exists():
            return load_inria_cameras_json(path / "cameras.json", image_dir=image_dir)
    raise ValueError(f"Don't know how to load cameras from {path}")


# ------------------------------------------------------------------ splits/images


def split_indices(n, test_every=8):
    """Standard 3DGS split on name-sorted cameras: every `test_every`-th is test."""
    test = [i for i in range(n) if i % test_every == 0]
    train = [i for i in range(n) if i % test_every != 0]
    return train, test


def evenly_spaced(indices, k):
    """Pick k indices spread evenly through `indices` (e.g. fitness views)."""
    indices = list(indices)
    if k >= len(indices):
        return indices
    return [indices[round(i * (len(indices) - 1) / (k - 1))] for i in range(k)] if k > 1 else [indices[0]]


def load_image(path, background=(0.0, 0.0, 0.0), size=None):
    """
    Load a ground-truth photo as float32 (H, W, 3) in [0, 1].
    RGBA images (NeRF-Synthetic) are composited onto `background`, which must
    match the background used when rendering. `size` = (W, H) resizes if needed.
    """
    from PIL import Image
    with Image.open(path) as im:
        if size is not None and im.size != tuple(size):
            im = im.resize(tuple(size), Image.LANCZOS)
        arr = np.asarray(im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB"),
                         dtype=np.float32) / 255.0
    if arr.shape[-1] == 4:
        rgb, a = arr[..., :3], arr[..., 3:]
        arr = rgb * a + np.asarray(background, dtype=np.float32) * (1 - a)
    return torch.from_numpy(np.ascontiguousarray(arr))


# --------------------------------------------------------------- sanity checking


@torch.no_grad()
def frustum_check(means, cams, max_points=200_000, near=0.01, seed=0):
    """
    Fraction of (a random sample of) Gaussian centers that are in front of each
    camera and project inside its image. Returns a (C,) tensor.

    Typical real scenes: tens of percent per camera. ~0% for most cameras means
    the poses are in the wrong convention (OpenGL vs OpenCV, c2w vs w2c), or the
    cameras don't belong to this scene.
    """
    means = means.float().cpu()
    if means.shape[0] > max_points:
        g = torch.Generator().manual_seed(seed)
        means = means[torch.randperm(means.shape[0], generator=g)[:max_points]]
    vm, K = cams.viewmats.cpu(), cams.Ks.cpu()
    pts = torch.einsum("cij,nj->cni", vm[:, :3, :3], means) + vm[:, None, :3, 3]   # (C, N, 3)
    z = pts[..., 2]
    uv = torch.einsum("cij,cnj->cni", K, pts)
    u, v = uv[..., 0] / z.clamp(min=1e-8), uv[..., 1] / z.clamp(min=1e-8)
    W = torch.tensor(cams.widths, dtype=torch.float32)[:, None]
    H = torch.tensor(cams.heights, dtype=torch.float32)[:, None]
    inside = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return inside.float().mean(dim=1)


def summarize(cams, means=None):
    train, test = split_indices(len(cams))
    K0 = cams.Ks[0]
    c = cams.centers()
    print(cams)
    print(f"Split (every 8th = test): {len(train)} train / {len(test)} test")
    print(f"First camera: {cams.names[0]}  {cams.widths[0]}x{cams.heights[0]}  "
          f"fx={K0[0,0]:.1f} fy={K0[1,1]:.1f} cx={K0[0,2]:.1f} cy={K0[1,2]:.1f}")
    print(f"Camera centers: min {c.min(0).values.numpy().round(2)}  max {c.max(0).values.numpy().round(2)}")
    missing = sum(p is None for p in cams.image_paths)
    if cams.image_paths:
        print(f"Ground-truth images found: {len(cams) - missing}/{len(cams)}")
    if means is not None:
        frac = frustum_check(means, cams)
        print(f"Frustum check: median {frac.median():.1%} of Gaussian centers visible per camera "
              f"(min {frac.min():.1%}, max {frac.max():.1%})")
        if frac.median() < 0.05:
            print("  !! Very low — likely a camera-convention bug or mismatched scene/cameras.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="COLMAP data dir, Inria cameras.json, or transforms json/dir")
    ap.add_argument("--images", default="images", help="image folder for COLMAP, e.g. images_4")
    ap.add_argument("--split", default="train", help="Blender split (train/test/val)")
    ap.add_argument("--image-dir", default=None, help="GT image folder for cameras.json")
    ap.add_argument("--scene", default=None, help="point_cloud.ply / .pt for the frustum check")
    args = ap.parse_args()

    cams = load_cameras(args.path, images=args.images, split=args.split, image_dir=args.image_dir)
    means = None
    if args.scene:
        from easplatting.gs_loader import load_scene
        means = load_scene(args.scene)["means"]
    summarize(cams, means)
