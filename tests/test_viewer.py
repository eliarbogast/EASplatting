"""viewer.py helpers: slider bounds, threshold masks, and render modes (fake rasterizer, CPU)."""

import math

import pytest
import torch

from conftest import H, W
from gs_loader import activate
from viewer import ViewerState, gene_stats, keep_mask, render_view, slider_bounds


@pytest.fixture()
def state(raw_scene):
    return ViewerState(raw_scene, activate(raw_scene), "cpu")


def test_slider_ends_keep_everything(state):
    b = state.bounds
    state.set_log_thresholds(b["min_opacity"][0], b["max_scale"][1], b["max_anisotropy"][1])
    assert state.mask.all()
    assert state.thresholds == dict(min_opacity=0.0, max_scale=float("inf"), max_anisotropy=float("inf"))


def test_mask_is_and_of_conditions_in_activated_space(state):
    act, st = state.act, gene_stats(state.act)
    lo, hi = slider_bounds(st)["min_opacity"]
    assert lo < hi
    state.set_log_thresholds(math.log10(0.2), state.bounds["max_scale"][1], state.bounds["max_anisotropy"][1])
    assert torch.equal(state.mask, act["opacities"] >= 0.2)

    ms_med = float(act["scales"].amax(-1).median())
    expect = (act["opacities"] >= 0.2) & (act["scales"].amax(-1) <= ms_med)
    state.set_log_thresholds(math.log10(0.2), math.log10(ms_med), state.bounds["max_anisotropy"][1])
    assert torch.equal(state.mask, expect)
    assert torch.equal(keep_mask(st, 0.2, ms_med, float("inf")), expect)


def test_splats_cache_rebuilds_only_when_mask_changes(state):
    m1, kept1, _ = state.splats()
    assert state.splats()[1] is kept1
    state.set_log_thresholds(math.log10(0.5), state.bounds["max_scale"][1], state.bounds["max_anisotropy"][1])
    m2, kept2, removed2 = state.splats()
    assert m2 is not m1 and kept2["means"].shape[0] + removed2["means"].shape[0] == state.n_total


@pytest.mark.parametrize("mode", ["pruned", "unpruned", "removed only", "error x5"])
def test_render_modes(state, mode):
    state.set_log_thresholds(math.log10(0.5), state.bounds["max_scale"][1], state.bounds["max_anisotropy"][1])
    state.view_mode = mode
    viewmat = torch.eye(4)
    viewmat[2, 3] = 3.0
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    img, info = render_view(state, viewmat, K, W, H)
    assert img.shape == (H, W, 3) and img.min() >= 0 and img.max() <= 1
    assert info["n_kept"] == int(state.mask.sum())
    if mode in ("pruned", "error x5"):
        assert info["psnr"] is not None and info["psnr"] < 99
    else:
        assert info["psnr"] is None
