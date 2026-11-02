"""
Fake `gsplat` for CPU-only tests (no CUDA needed).

Implements `rasterization` with the same signature/outputs the pipeline uses,
as a crude point splat: each visible Gaussian adds opacity * color at the pixel
its center projects to. It is LINEAR in colors (like the real rasterizer), so the
blend-weight gradient trick in render.gaussian_contributions is testable:
expected blend_weight = opacity * (number of views where it is visible).

It also asserts that inputs are ACTIVATED (normalized quats, positive scales,
opacities in [0, 1]) — catching a missed gs_loader.activate().

Tests put tests/fake_gsplat on sys.path / PYTHONPATH so `import gsplat` resolves here.
"""

import torch

__version__ = "fake-0.0"


def rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                  near_plane=0.01, far_plane=1e10, sh_degree=None, backgrounds=None,
                  packed=True, rasterize_mode="classic", **kwargs):
    C, N = viewmats.shape[0], means.shape[0]
    assert quats.shape == (N, 4) and scales.shape == (N, 3) and opacities.shape == (N,)
    assert torch.allclose(quats.norm(dim=-1), torch.ones(N, dtype=quats.dtype), atol=1e-4), "quats not normalized"
    assert (scales > 0).all(), "scales must be exp()-activated"
    assert ((opacities >= 0) & (opacities <= 1)).all(), "opacities must be sigmoid()-activated"
    if sh_degree is not None:
        assert colors.shape == (N, (sh_degree + 1) ** 2, 3), f"bad SH shape {tuple(colors.shape)}"
        col = 0.5 + 0.28209479177387814 * colors[:, 0, :]
    else:
        col = colors
    D = col.shape[-1]

    pts = torch.einsum("cij,nj->cni", viewmats[:, :3, :3], means) + viewmats[:, None, :3, 3]
    z = pts[..., 2]
    uv = torch.einsum("cij,cnj->cni", Ks, pts)
    u = torch.floor(uv[..., 0] / z.clamp(min=1e-8)).long()
    v = torch.floor(uv[..., 1] / z.clamp(min=1e-8)).long()
    vis = (z > near_plane) & (z < far_plane) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    img = torch.zeros(C, height * width, D, dtype=col.dtype)
    alpha = torch.zeros(C, height * width, 1)
    for c in range(C):
        idx = (v[c] * width + u[c])[vis[c]]
        img[c] = img[c].index_add(0, idx, (opacities[:, None] * col)[vis[c]])
        alpha[c] = alpha[c].index_add(0, idx, opacities[vis[c], None].detach())
    img = img.view(C, height, width, D)
    alpha = alpha.view(C, height, width, 1).clamp(max=1)
    if backgrounds is not None:
        # real gsplat 1.5.3 rejects per-camera backgrounds when packed=True (shape check vs packed dims)
        assert not packed, "gsplat 1.5.3 asserts on `backgrounds` with packed=True"
        img = img + backgrounds[:, None, None, :] * (1 - alpha)
    radii = (vis.long() * 3)[..., None].expand(C, N, 2).to(torch.int32)
    return img, alpha, {"radii": radii}
