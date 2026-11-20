import numpy as np
import torch

from easplatting.explore import local_rank, make_mask_fn, rank, warm_genome


def test_rank_is_a_percentile():
    r = rank(torch.tensor([5.0, 1.0, 3.0, 2.0]))
    assert r.tolist() == [0.75, 0.0, 0.5, 0.25]


def test_fixed_weights_one_region_is_the_1d_sweep():
    feats = torch.stack([rank(torch.rand(1000)), rank(torch.rand(1000))], dim=1)
    regions = torch.zeros(1000, dtype=torch.long)
    mask_fn, n = make_mask_fn(feats, regions, 1, False, fixed_feature=0)
    assert n == 1
    mask = mask_fn(np.array([0.4]))
    assert torch.equal(mask, feats[:, 0] >= 0.4)
    assert mask_fn(np.array([0.0])).all()  # the explicit off zone keeps everything


def test_learned_weights_nest_the_baseline_via_the_warm_genome():
    feats = torch.stack([rank(torch.rand(1000)) for _ in range(3)], dim=1)
    regions = torch.zeros(1000, dtype=torch.long)
    mask_fn, n = make_mask_fn(feats, regions, 1, True, fixed_feature=1)
    warm = warm_genome(3, 1, True, 1, 0.45)
    assert n == 4 and len(warm) == 4
    # weights (0, +1, 0) rank the same as feature 1 alone
    assert torch.equal(mask_fn(warm), feats[:, 1] >= 0.45)


def test_regions_get_their_own_cutoffs_and_local_rank_is_per_region():
    score = torch.rand(1000)
    regions = (torch.arange(1000) % 2).long()
    feats = rank(score)[:, None]
    mask_fn, n = make_mask_fn(feats, regions, 2, False, fixed_feature=0)
    mask = mask_fn(np.array([0.0, 0.5]))
    assert mask[regions == 0].all()
    assert not torch.equal(mask[regions == 1], torch.ones(500, dtype=torch.bool))
    lr = local_rank(score, regions, 2)
    assert abs(lr[regions == 1].mean().item() - 0.5) < 0.01
    local_fn, _ = make_mask_fn(feats, regions, 2, False, 0, local=True)
    assert abs(local_fn(np.array([0.3, 0.3])).float().mean().item() - 0.7) < 0.01


def test_average_ties_gives_tied_values_the_same_rank():
    x = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 6.0])
    r = rank(x, average_ties=True)
    assert torch.allclose(r, torch.tensor([1.5, 1.5, 1.5, 1.5, 4.0, 5.0]) / 6)
    # without ties both rankings agree
    y = torch.rand(100)
    assert torch.equal(rank(y, average_ties=True), rank(y))
