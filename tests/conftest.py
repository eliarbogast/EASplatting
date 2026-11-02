"""
Shared fixtures. Tests run on CPU with a fake gsplat (tests/fake_gsplat), so they
work in CI / cloud sessions without a GPU. They check plumbing — formats,
conventions, batching, caching, metrics, reports — not real rendering quality.
Real rendering is checked by scripts/setup_local.sh's smoke test and evaluate.py.
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).resolve().parent / "fake_gsplat"
for p in (str(FAKE), str(ROOT)):           # fake gsplat first, then the project modules
    if p not in sys.path:
        sys.path.insert(0, p)


def lookat_c2w_cv(eye, target=np.zeros(3), up=np.array([0.0, 0.0, 1.0])):
    """Camera->world in OpenCV convention (x right, y down, z forward) looking at target."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([r, d, f], 1)
    c2w[:3, 3] = eye
    return c2w


def rot_to_quat(R):
    """Rotation matrix -> (w, x, y, z), numerically robust (Shepperd)."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4])


N_CAMS, W, H, F = 16, 160, 120, 140.0


@pytest.fixture(scope="session")
def ring_c2ws():
    return [lookat_c2w_cv(np.array([4 * math.cos(a), 4 * math.sin(a), 1.0]))
            for a in np.linspace(0, 2 * math.pi, N_CAMS, endpoint=False)]


@pytest.fixture(scope="session")
def raw_scene():
    """Random raw (pre-activation) scene clustered at the origin."""
    torch.manual_seed(0)
    n = 4000
    return dict(means=torch.randn(n, 3) * 0.6, quats=torch.randn(n, 4), scales=torch.randn(n, 3) - 4,
                opacities=torch.randn(n) * 3, sh0=torch.randn(n, 1, 3) * 0.5, shN=torch.randn(n, 15, 3) * 0.1)


def write_colmap(scene_dir, c2ws, factor_dirs=("images",), text=False):
    """Write a PINHOLE COLMAP model (bin or txt) + blank images; names written out of order."""
    from PIL import Image
    sp = scene_dir / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = F, F * 1.01, W / 2 + 3, H / 2 - 2
    names = [f"img_{i:03d}.jpg" for i in range(len(c2ws))][::-1]
    if text:
        (sp / "cameras.txt").write_text(f"# c\n1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
        lines = ["# images"]
        for k, nm in enumerate(names):
            i = int(nm[4:7])
            w2c = np.linalg.inv(c2ws[i])
            q, t = rot_to_quat(w2c[:3, :3]), w2c[:3, 3]
            lines.append(f"{k} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {nm}")
            lines.append("1.0 2.0 -1 3.0 4.0 5" if k % 2 == 0 else "")   # empty points line too
        (sp / "images.txt").write_text("\n".join(lines) + "\n")
    else:
        with open(sp / "cameras.bin", "wb") as fh:
            fh.write(struct.pack("<Q", 1))
            fh.write(struct.pack("<iiQQ", 1, 1, W, H))
            fh.write(struct.pack("<4d", fx, fy, cx, cy))
        with open(sp / "images.bin", "wb") as fh:
            fh.write(struct.pack("<Q", len(names)))
            for k, nm in enumerate(names):
                i = int(nm[4:7])
                w2c = np.linalg.inv(c2ws[i])
                fh.write(struct.pack("<i", k))
                fh.write(struct.pack("<4d", *rot_to_quat(w2c[:3, :3])))
                fh.write(struct.pack("<3d", *w2c[:3, 3]))
                fh.write(struct.pack("<i", 1))
                fh.write(nm.encode() + b"\x00")
                fh.write(struct.pack("<Q", 2))
                fh.write(b"\x00" * 48)
    for d in factor_dirs:
        f = 1 if d == "images" else int(d.split("_")[1])
        (scene_dir / d).mkdir(exist_ok=True)
        for i in range(len(c2ws)):
            Image.new("RGB", (W // f, H // f)).save(scene_dir / d / f"img_{i:03d}.jpg")
    return dict(fx=fx, fy=fy, cx=cx, cy=cy)


@pytest.fixture()
def colmap_scene(tmp_path, ring_c2ws):
    d = tmp_path / "colmap"
    intr = write_colmap(d, ring_c2ws, factor_dirs=("images", "images_2"))
    return d, intr


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj))
