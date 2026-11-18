"""learned gates and the Fisher score on the fake rasterizer (CPU)."""

import numpy as np
import torch
from PIL import Image

from conftest import H, W
from easplatting.cameras import Cameras
from easplatting.gs_loader import activate
from easplatting.render import render_views
from easplatting.scores import fisher_scores, learned_gates, load_photos


def toy_cameras(count=4):
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    viewmats = torch.eye(4).repeat(count, 1, 1)
    viewmats[:, 2, 3] = 3.0
    viewmats[:, 0, 3] = torch.linspace(0, 0.5, count)
    return Cameras(
        names=[f"v{i}" for i in range(count)],
        viewmats=viewmats,
        Ks=K.expand(count, 3, 3).clone(),
        widths=[W] * count,
        heights=[H] * count,
    )


def test_learned_gates_rank_visible_above_invisible(raw_scene):
    torch.manual_seed(0)
    scene = activate(raw_scene)
    cams = toy_cameras()
    baselines = render_views(scene, cams)
    seen = render_views(scene, cams).sum() > 0
    logits = learned_gates(scene, cams, baselines, steps=120, batch_size=4)

    # gaussians that never reach a pixel only pay the price of their gate
    stats = scene["means"].shape[0]
    visible = torch.zeros(stats, dtype=torch.bool)
    for i in range(len(cams)):
        point = torch.einsum("ij,nj->ni", cams.viewmats[i, :3, :3], scene["means"])
        point = point + cams.viewmats[i, :3, 3]
        pixel = torch.einsum("ij,nj->ni", cams.Ks[i], point)
        u = torch.floor(pixel[:, 0] / point[:, 2].clamp(min=1e-8))
        v = torch.floor(pixel[:, 1] / point[:, 2].clamp(min=1e-8))
        visible |= (point[:, 2] > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    assert seen and logits.shape == (stats,)
    assert visible.any() and (~visible).any()
    assert logits[visible].mean() > logits[~visible].mean()  # the useful ones stay open


def test_fisher_scores_are_zero_for_gaussians_nobody_sees(raw_scene):
    scene = activate(raw_scene)
    cams = toy_cameras(3)
    photos = render_views(scene, cams) * 0.9  # photos that differ from the model
    score = fisher_scores(scene, cams, photos)
    seen = score > 0
    assert seen.any() and (score >= 0).all()
    assert (score[~seen] == 0).all()


def test_load_photos_resizes_to_the_view_resolution(tmp_path):
    cams = toy_cameras(2)
    paths = []
    for i in range(2):
        paths.append(str(tmp_path / f"{i}.png"))
        Image.fromarray(np.full((H * 2, W * 2, 3), 100 + i, dtype=np.uint8)).save(paths[-1])
    cams.image_paths = paths
    photos = load_photos(cams, "cpu")
    assert photos.shape == (2, H, W, 3) and abs(photos[1].mean().item() - 101 / 255) < 1e-2
