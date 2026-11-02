"""render.py, metrics.py and evaluate.py end-to-end — with the fake gsplat, on CPU."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cameras import load_colmap
from conftest import FAKE, H, ROOT, W
from gs_loader import activate, load_scene, prune, save_ply
from render import (cache_baselines, gaussian_contributions, load_baselines, psnr, render_views,
                    save_png, select_indices)


@pytest.fixture()
def scene_and_cams(tmp_path, colmap_scene, raw_scene):
    d, _ = colmap_scene
    save_ply(raw_scene, tmp_path / "scene.ply")
    return tmp_path / "scene.ply", load_colmap(d, images="images_2"), d


def test_render_batching_and_empty_scene(scene_and_cams):
    ply, cams, _ = scene_and_cams
    full = activate(load_scene(ply))
    idx = select_indices(len(cams), "train", every=2)
    imgs = render_views(full, cams, idx, batch_size=3)
    assert imgs.shape == (len(idx), H // 2, W // 2, 3) and 0 <= imgs.min() and imgs.max() <= 1
    assert torch.allclose(imgs, render_views(full, cams, idx, batch_size=1))     # batching invariant
    empty = render_views(prune(full, torch.zeros(4000, dtype=torch.bool)), cams, idx[:2], background=(1, 1, 1))
    assert (empty == 1).all()


def test_baseline_cache_roundtrip(tmp_path, scene_and_cams):
    ply, cams, _ = scene_and_cams
    idx = select_indices(len(cams), "train", every=2)
    cache_baselines(ply, cams, idx, tmp_path / "cache" / "b.pt", device="cpu", batch_size=4)
    b = load_baselines(tmp_path / "cache" / "b.pt", device="cpu")
    ref = render_views(activate(load_scene(ply)), cams, idx)
    assert b["indices"] == idx and b["num_gaussians"] == 4000
    assert psnr(b["images"], ref) > 60                                            # fp16 error is negligible


def test_contributions_match_fake_rasterizer(scene_and_cams):
    ply, cams, _ = scene_and_cams
    full = activate(load_scene(ply))
    idx = select_indices(len(cams), "train", every=2)
    c = gaussian_contributions(full, cams, idx, batch_size=4)
    # fake rasterizer: weight per visible view == opacity
    assert torch.allclose(c["blend_weight"], full["opacities"] * c["visible_views"].float(), atol=1e-5)
    assert c["visible_views"].max() <= len(idx)


def test_metrics_basic():
    pytest.importorskip("torchmetrics")
    from metrics import ImageMetrics, psnr_fast, summarize
    torch.manual_seed(0)
    x = torch.rand(3, 48, 48, 3)
    y = (x + 0.01).clamp(0, 1)
    m = ImageMetrics(device="cpu", psnr_cap=60)
    r = m(x, y, lpips=False)
    assert len(r["psnr"]) == 3 and 38 < r["psnr"][0] < 42 and r["ssim"][0] > 0.95
    assert summarize(m(x, x, lpips=False))["psnr"] == 60                          # capped, not inf
    assert torch.allclose(psnr_fast(x, y), torch.tensor(r["psnr"]), atol=0.05)


def test_evaluate_end_to_end(tmp_path, scene_and_cams):
    """Full evaluate.py run: GT = fake renders saved as PNG, so metrics must be near-perfect."""
    pytest.importorskip("torchmetrics")
    ply, cams, d = scene_and_cams
    full = activate(load_scene(ply))
    gt_dir = d / "images_2"
    imgs = render_views(full, cams)                                               # all views
    for j, name in enumerate(cams.names):
        p = gt_dir / name
        save_png(imgs[j], p.with_suffix(".png"))
        os.replace(p.with_suffix(".png"), p)                                      # keep the .jpg name, PNG bytes
    nb = tmp_path / "nb.json"
    nb.write_text(json.dumps({"metrics": {"psnr": 50.0, "ssim": 1.0, "lpips_vgg": 0.0}}))

    out = tmp_path / "eval"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(FAKE), str(ROOT)]))
    cmd = [sys.executable, str(ROOT / "evaluate.py"), "--device", "cpu", "--scene", str(ply),
           "--cameras", str(d), "--images", "images_2", "--split", "test", "--no-lpips",
           "--prune-opacity", "0.5", "--cache-baselines", "--nb-metrics", str(nb), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((out / "report.json").read_text())
    assert rep["env"]["gsplat"].startswith("fake")
    assert rep["checks"]["frustum_median_visible"] > 0.5 and rep["checks"]["psnr_sane"]
    assert rep["checks"]["gt_size_matches_camera"]
    assert rep["metrics_vs_gt"]["psnr"] > 45                                      # only uint8 quantization error
    assert 0 < rep["prune_demo"]["kept_frac"] < 1
    assert "nerfbaselines" in rep["crosschecks"]
    assert (out / "baselines_test.pt").exists() and len(list((out / "compare").glob("*.png"))) == 2
    rows = list(csv.DictReader(open(out / "per_view.csv")))
    assert len(rows) == 2 and {"psnr", "ssim", "pruned_psnr"} <= set(rows[0])


def test_evaluate_refuses_misaligned_cameras(tmp_path, scene_and_cams):
    """Scene shifted far away (like a gsplat-normalized scene vs raw COLMAP) -> frustum check fails."""
    ply, _, d = scene_and_cams
    raw = load_scene(ply)
    raw["means"] = raw["means"] + 100.0
    save_ply(raw, tmp_path / "far.ply")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(FAKE), str(ROOT)]))
    r = subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--device", "cpu", "--scene",
                        str(tmp_path / "far.ply"), "--cameras", str(d), "--images", "images_2",
                        "--no-lpips", "--out", str(tmp_path / "e")], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "different frames" in r.stdout


@pytest.mark.parametrize("lpips_ref, lpips_net, ok", [
    (0.2538, "vgg", True),        # within 0.01 of ours (0.2622 would be the real-bonsai value: 0.008 off)
    (0.30, "vgg", False),         # LPIPS drift now fails the check
    (0.30, "vgg-raw", True),      # raw (Inria) convention is not comparable, so LPIPS is skipped
])
def test_crosscheck_enforces_lpips(tmp_path, lpips_ref, lpips_net, ok):
    from evaluate import crosscheck
    ref = tmp_path / "nb.json"
    ref.write_text(json.dumps({"metrics": {"psnr": 30.0, "ssim": 0.9, "lpips_vgg": lpips_ref}}))
    report = {"metrics_vs_gt": {"psnr": 30.01, "ssim": 0.901, "lpips": 0.2538}, "checks": {}}
    crosscheck(report, ref, "nerfbaselines", lpips_net, True, hint="", psnr_tol=0.1, ssim_tol=0.003)
    assert report["checks"]["matches_nerfbaselines"] is ok
