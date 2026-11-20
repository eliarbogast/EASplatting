"""features.py — extra per-Gaussian features for the explore.py genome (each becomes a percentile rank there).

view_features   one pass over the train views (half resolution), per Gaussian:
    tailq / tailmin  10th percentile / minimum over the views that see it of log10 blend weight (a view where it is occluded
                     counts: the tail, not the best view, which is what `max` rewards)
    edge             mean image-gradient strength (Sobel of the unpruned render, per view normalized) at its projected centre
    foot             mean log of its screen radius in pixels (sub-pixel Gaussians matter differently at other resolutions)
    depth            mean log camera depth when visible
neighbor_counts number of other Gaussian centres within `factor` x its own largest scale (redundancy; CPU, scipy KD-tree)
"""

import numpy as np
import torch
import torch.nn.functional as F

from easplatting.render import _rasterize, render_views


def sobel_magnitude(image):
    """(H, W, 3) image in [0, 1] -> (H, W) gradient magnitude of its gray version, divided by its mean."""
    gray = image.mean(dim=-1)[None, None]
    kernel = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device)
    gx = F.conv2d(gray, kernel[None, None], padding=1)
    gy = F.conv2d(gray, kernel.T[None, None], padding=1)
    magnitude = (gx**2 + gy**2).sqrt()[0, 0]
    return magnitude / magnitude.mean().clamp(min=1e-8)


def project(means, viewmat, K):
    """Pixel coordinates (N, 2) and camera depth (N,) of world points (OpenCV world -> camera)."""
    cam = means @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = cam[:, 2]
    safe = z.clamp(min=1e-6)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return torch.stack([u, v], dim=1), z


def sample_map(values, uv, width, height):
    """Bilinear samples of an (H, W) map at pixel coordinates (N, 2) -> (N,)."""
    grid = torch.stack(
        [2 * uv[:, 0] / (width - 1) - 1, 2 * uv[:, 1] / (height - 1) - 1], dim=-1
    )
    grid = grid[None, :, None, :]
    out = F.grid_sample(
        values[None, None], grid, align_corners=True, padding_mode="border"
    )
    return out[0, 0, :, 0]


def nan_percentile(values, q, chunk=100_000, fill=-9.0):
    """Row-wise percentile (0..1) of an (N, V) matrix ignoring NaN; all-NaN rows get `fill`."""
    out = torch.empty(len(values), device=values.device)
    for start in range(0, len(values), chunk):
        part = values[start : start + chunk].float()
        out[start : start + chunk] = torch.nanquantile(part, q, dim=1)
    return torch.nan_to_num(out, nan=fill)


def nan_min(values, chunk=100_000, fill=-9.0):
    """Row-wise minimum of an (N, V) matrix ignoring NaN; all-NaN rows get `fill`."""
    out = torch.empty(len(values), device=values.device)
    for start in range(0, len(values), chunk):
        part = values[start : start + chunk].float()
        out[start : start + chunk] = torch.where(
            torch.isnan(part), torch.full_like(part, float("inf")), part
        ).amin(dim=1)
    return torch.where(torch.isinf(out), torch.full_like(out, fill), out)


def view_features(scene, views):
    """scene: activated scene, views: Cameras (one image size). Returns a dict of (N,) CPU tensors."""
    device = scene["means"].device
    count = scene["means"].shape[0]
    width, height = views.size
    viewmats, Ks = views.viewmats.to(device), views.Ks.to(device)
    detached = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in scene.items()}

    log_weights = torch.full((count, len(views)), float("nan"), dtype=torch.float16, device=device)
    seen = torch.zeros(count, device=device)
    seen_in_image = torch.zeros(count, device=device)
    foot_sum = torch.zeros(count, device=device)
    depth_sum = torch.zeros(count, device=device)
    edge_sum = torch.zeros(count, device=device)

    for i in range(len(views)):
        # the edge map comes from the unpruned render of this view
        render = render_views(scene, views, [i], batch_size=1)[0]
        edges = sobel_magnitude(render)

        # blend weights and screen radii of every Gaussian in this view (one backward pass)
        ones = torch.ones(count, 1, device=device, requires_grad=True)
        image, _, meta = _rasterize(
            detached,
            viewmats[i : i + 1],
            Ks[i : i + 1],
            width,
            height,
            colors=ones,
            sh_degree=None,
            packed=False,
        )
        image.sum().backward()
        weight = ones.grad[:, 0]
        radii = meta["radii"][0]
        radii = radii[:, None] if radii.ndim == 1 else radii
        visible = (radii > 0).all(dim=-1)

        with torch.no_grad():
            log_weight = torch.log10(weight.clamp(min=1e-8))
            log_weights[:, i] = torch.where(
                visible, log_weight, torch.full_like(log_weight, float("nan"))
            ).half()
            seen += visible.float()
            foot_sum += torch.where(
                visible, torch.log(radii.float().amax(dim=-1).clamp(min=0.5)), torch.zeros_like(weight)
            )
            uv, z = project(scene["means"], viewmats[i], Ks[i])
            depth_sum += torch.where(visible, torch.log(z.clamp(min=1e-6)), torch.zeros_like(weight))
            inside = visible & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
            edge_sum += torch.where(inside, sample_map(edges, uv, width, height), torch.zeros_like(weight))
            seen_in_image += inside.float()

    return dict(
        tailq=nan_percentile(log_weights, 0.1).cpu(),
        tailmin=nan_min(log_weights).cpu(),
        edge=(edge_sum / seen_in_image.clamp(min=1)).cpu(),
        foot=(foot_sum / seen.clamp(min=1)).cpu(),
        depth=(depth_sum / seen.clamp(min=1)).cpu(),
    )


def neighbor_counts(means, scales, factor=3.0, cap_quantile=0.95):
    """Other Gaussian centres within factor x the Gaussian's largest scale (radius capped so giant Gaussians stay cheap)."""
    from scipy.spatial import cKDTree

    points = means.detach().cpu().numpy()
    radius = factor * scales.detach().amax(dim=-1).cpu().numpy()
    radius = np.minimum(radius, np.quantile(radius, cap_quantile))
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True, workers=-1) - 1
    return torch.from_numpy(counts.astype(np.float32))
