"""Scene + camera formats and conventions (gs_loader.py, cameras.py, export format)."""

import json
import math
import warnings

import numpy as np
import pytest
import torch
from PIL import Image

from cameras import Cameras, evenly_spaced, frustum_check, load_cameras, load_colmap, load_image, split_indices
from conftest import F, H, N_CAMS, W, write_colmap
from gs_loader import activate, bytes_per_gaussian, load_ply, load_scene, prune, save_ply


# ------------------------------------------------------------------ scenes

def test_ply_roundtrip_and_activation(tmp_path, raw_scene):
    save_ply(raw_scene, tmp_path / "s.ply")
    back = load_ply(tmp_path / "s.ply")
    for k in raw_scene:
        assert torch.allclose(raw_scene[k], back[k]), k
    a = activate(back)
    assert a["sh_degree"] == 3 and a["colors"].shape == (4000, 16, 3)
    assert torch.allclose(a["quats"].norm(dim=-1), torch.ones(4000), atol=1e-5)
    assert (a["opacities"] > 0).all() and (a["opacities"] < 1).all()
    assert bytes_per_gaussian(back) == 248


def test_ply_sh_layout_is_channel_major(tmp_path, raw_scene):
    """f_rest_0..14 must be the R channel of SH bands 1..15 (Inria / gsplat convention)."""
    save_ply(raw_scene, tmp_path / "s.ply")
    from plyfile import PlyData
    v = PlyData.read(str(tmp_path / "s.ply"))["vertex"]
    assert np.allclose(v["f_rest_0"], raw_scene["shN"][:, 0, 0].numpy())   # band 1, R
    assert np.allclose(v["f_rest_1"], raw_scene["shN"][:, 1, 0].numpy())   # band 2, R
    assert np.allclose(v["f_rest_15"], raw_scene["shN"][:, 0, 1].numpy())  # band 1, G


def test_gsplat_ckpt_and_sh0_scene(tmp_path, raw_scene):
    torch.save({"step": 6999, "splats": raw_scene}, tmp_path / "ckpt.pt")
    c = load_scene(tmp_path / "ckpt.pt")
    assert torch.allclose(c["shN"], raw_scene["shN"])
    raw0 = dict(raw_scene, shN=torch.zeros(4000, 0, 3))
    save_ply(raw0, tmp_path / "s0.ply")
    assert activate(load_ply(tmp_path / "s0.ply"))["sh_degree"] == 0


def test_prune_keeps_non_tensor_fields(raw_scene):
    a = activate(raw_scene)
    m = a["opacities"] > 0.5
    p = prune(a, m)
    assert p["means"].shape[0] == int(m.sum()) and p["sh_degree"] == 3


# ----------------------------------------------------------------- cameras

def test_colmap_bin_and_txt_agree(tmp_path, ring_c2ws):
    write_colmap(tmp_path / "b", ring_c2ws, factor_dirs=("images", "images_2"))
    write_colmap(tmp_path / "t", ring_c2ws, factor_dirs=("images", "images_2"), text=True)
    cb, ct = load_colmap(tmp_path / "b", images="images_2"), load_colmap(tmp_path / "t", images="images_2")
    assert cb.names == sorted(cb.names) and len(cb) == N_CAMS
    assert torch.allclose(cb.viewmats, ct.viewmats, atol=1e-5) and torch.allclose(cb.Ks, ct.Ks)
    # images_2: intrinsics scaled by the ACTUAL image size
    assert cb.size == (W // 2, H // 2)
    assert abs(cb.Ks[0, 0, 0] - F / 2) < 1e-3 and abs(cb.Ks[0, 0, 2] - (W / 2 + 3) / 2) < 1e-3
    # poses: world->camera, OpenCV, matching what we wrote
    assert torch.allclose(cb.viewmats[0], torch.tensor(np.linalg.inv(ring_c2ws[0])).float(), atol=1e-5)
    assert torch.allclose(cb.centers()[3], torch.tensor(ring_c2ws[3][:3, 3]).float(), atol=1e-5)


def test_colmap_missing_image_folder_falls_back(colmap_scene):
    d, _ = colmap_scene
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = load_colmap(d, images="images_4")
    assert c.size == (W // 4, H // 4) and abs(c.Ks[0, 0, 0] - F / 4) < 1e-3
    assert any("No images" in str(x.message) for x in w)


def test_frustum_check_detects_wrong_convention(colmap_scene, ring_c2ws):
    d, _ = colmap_scene
    pts = torch.randn(20000, 3) * 0.6
    good = load_colmap(d, images="images_2")
    assert frustum_check(pts, good).min() > 0.9
    # Using camera->world as if it were world->camera (a classic bug) -> nothing visible
    bad = Cameras(good.names, torch.tensor(np.stack(ring_c2ws)).float(), good.Ks, good.widths, good.heights)
    assert frustum_check(pts, bad).max() < 0.05


def test_inria_and_exported_cameras_json(tmp_path, colmap_scene, ring_c2ws):
    d, intr = colmap_scene
    ref = load_colmap(d, images="images")
    entries = []
    for i, c2w in enumerate(ring_c2ws):
        e = dict(id=i, img_name=f"img_{i:03d}", width=W, height=H, position=c2w[:3, 3].tolist(),
                 rotation=c2w[:3, :3].tolist(), fx=intr["fx"], fy=intr["fy"])
        entries.append(e)
    # plain Inria file: principal point assumed at center, images found via --image-dir
    (tmp_path / "inria").mkdir()
    (tmp_path / "inria" / "cameras.json").write_text(json.dumps(entries[::-1]))
    ci = load_cameras(tmp_path / "inria" / "cameras.json", image_dir=d / "images")
    assert torch.allclose(ci.viewmats, ref.viewmats, atol=1e-5)
    assert ci.image_paths[0] is not None and ci.Ks[0, 0, 2] == W / 2
    # exported (export_gsplat_cameras.py) file: explicit cx/cy + image_path
    for e in entries:
        e.update(cx=intr["cx"], cy=intr["cy"], image_path=str(d / "images" / f"{e['img_name']}.jpg"))
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "cameras.json").write_text(json.dumps(entries))
    ce = load_cameras(tmp_path / "exp" / "cameras.json")
    assert torch.allclose(ce.Ks, ref.Ks, atol=1e-4) and all(p is not None for p in ce.image_paths)


def test_blender_opengl_conversion(tmp_path, ring_c2ws):
    b = tmp_path / "blender"
    (b / "train").mkdir(parents=True)
    frames = []
    for i, c2w in enumerate(ring_c2ws):
        gl = c2w.copy()
        gl[:3, 1:3] *= -1                                    # OpenCV -> OpenGL
        frames.append(dict(file_path=f"./train/r_{i}", transform_matrix=gl.tolist()))
        Image.new("RGBA", (200, 200), (255, 0, 0, 128)).save(b / "train" / f"r_{i}.png")
    (b / "transforms_train.json").write_text(json.dumps(dict(camera_angle_x=2 * math.atan(100 / F), frames=frames)))
    c = load_cameras(b)
    assert c.size == (200, 200) and abs(c.Ks[0, 0, 0] - F) < 1e-3
    assert torch.allclose(c.viewmats, torch.tensor(np.stack([np.linalg.inv(m) for m in ring_c2ws])).float(), atol=1e-5)
    img = load_image(c.image_paths[0], background=(1, 1, 1))  # RGBA composited on white
    assert img.shape == (200, 200, 3) and abs(img[0, 0, 0] - 1) < 1e-2 and abs(img[0, 0, 1] - 127 / 255) < 0.01


def test_splits_match_llffhold8():
    tr, te = split_indices(16)
    assert te == [0, 8] and len(tr) == 14 and not set(tr) & set(te)
    assert len(evenly_spaced(tr, 4)) == 4 and evenly_spaced(tr, 1) == [tr[0]]
