import numpy as np
import torch

from easplatting.features import (
    nan_min,
    nan_percentile,
    neighbor_counts,
    project,
    sample_map,
    sobel_magnitude,
)


def test_sobel_finds_a_step_edge_and_is_mean_normalized():
    image = torch.zeros(16, 16, 3)
    image[:, 8:] = 1.0
    magnitude = sobel_magnitude(image)
    assert abs(magnitude.mean().item() - 1.0) < 1e-5
    assert magnitude[8, 7:9].min() > 5 * magnitude[8, :4].max().clamp(min=1e-6)


def test_project_and_sample_round_trip():
    K = torch.tensor([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    viewmat = torch.eye(4)
    points = torch.tensor([[0.0, 0.0, 2.0], [0.2, -0.1, 4.0]])
    uv, z = project(points, viewmat, K)
    assert torch.allclose(uv[0], torch.tensor([32.0, 24.0]))
    assert torch.allclose(uv[1], torch.tensor([37.0, 21.5]))
    assert torch.allclose(z, torch.tensor([2.0, 4.0]))
    # a map whose value is its own x coordinate returns the pixel x
    values = torch.arange(64.0)[None, :].repeat(48, 1)
    assert torch.allclose(sample_map(values, uv, 64, 48), uv[:, 0], atol=1e-4)


def test_nan_statistics_ignore_nan_and_fill_empty_rows():
    values = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [float("nan")] * 4, [5.0, float("nan"), 1.0, float("nan")]]
    )
    assert nan_min(values).tolist() == [1.0, -9.0, 1.0]
    q = nan_percentile(values, 0.5)
    assert q[0].item() == 2.5 and q[1].item() == -9.0 and q[2].item() == 3.0


def test_neighbor_counts_counts_centres_within_the_own_scale():
    means = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.0, 0.1, 0], [10.0, 10, 10]])
    scales = torch.full((4, 3), 0.1)
    counts = neighbor_counts(means, scales, factor=2.0, cap_quantile=1.0)
    # radius 0.2: the three clustered centres all lie within it of each other, the far one is alone
    assert counts.tolist() == [2.0, 2.0, 2.0, 0.0]
