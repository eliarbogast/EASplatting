"""
thresholds.py — the pruning genes as plain threshold -> keep-mask logic (CPU-testable, no rendering).

A Gaussian is KEPT when ALL conditions pass (activated values: sigmoid opacity, exp scale):
    opacity >= min_opacity  and  max(scale) <= max_scale  and  max(scale)/min(scale) <= max_anisotropy
    and  importance_rank >= min_importance      (optional 4th gene, see gene_stats)

"Off" for a gene means thresholds 0 / inf. It must be exactly reachable: `slider_bounds` stops the
anisotropy range at the 99.9th percentile (a few degenerate Gaussians have ratios up to 1e10), which
is NOT off: those ~0.1% extreme Gaussians are visible and removing them caps quality near 40 dB vs the
unpruned render. Viewers and genome decoders therefore need an explicit "off" zone at the range ends.

Used by viewer.py today and by the genome decoder next.
"""

import math

import torch


def gene_stats(act, importance=None):
    """Per-Gaussian quantities the sliders threshold (compute once per scene).
    importance: optional (N,) score per Gaussian (render.gaussian_contributions()["blend_weight"]);
    stored as its percentile rank in [0, 1) so the gene "remove the lowest p" is scene independent."""
    smax, smin = act["scales"].amax(-1), act["scales"].amin(-1)
    stats = dict(opacity=act["opacities"], max_scale=smax, anisotropy=smax / smin.clamp(min=1e-12))
    if importance is not None:
        rank = torch.empty(len(importance), device=importance.device)
        rank[importance.argsort()] = torch.arange(len(importance), device=importance.device, dtype=torch.float32)
        stats["importance"] = rank / len(importance)
    return stats


def _quantile(t, q, max_n=1_000_000):
    """torch.quantile with a stride subsample (it refuses tensors of more than ~16M elements)."""
    return float(torch.quantile(t.float()[:: max(1, len(t) // max_n)], q))


def slider_bounds(stats, margin=0.01):
    """
    log10 slider ranges derived from the scene's own distributions. The 'off' end of each slider
    keeps every Gaussian. Anisotropy has a few degenerate outliers (min scale ~ 0, ratio up to
    1e10) that would squash the slider, so its range stops at the 99.9th percentile; only the
    'off' end keeps those.
    """
    lg = lambda t: torch.log10(t.clamp(min=1e-12))
    op, ms, an = lg(stats["opacity"]), lg(stats["max_scale"]), lg(stats["anisotropy"])
    return dict(
        min_opacity=(float(op.min()) - margin, math.log10(0.5)),
        max_scale=(_quantile(ms, 0.5), float(ms.max()) + margin),
        max_anisotropy=(0.0, _quantile(an, 0.999) + margin),
    )


def keep_mask(stats, min_opacity, max_scale, max_anisotropy, min_importance=0.0):
    """Boolean keep-mask from ACTIVATED-space thresholds (linear values, not log10).
    min_importance is a percentile (0 = off); it needs stats from gene_stats(act, importance)."""
    mask = ((stats["opacity"] >= min_opacity) & (stats["max_scale"] <= max_scale)
            & (stats["anisotropy"] <= max_anisotropy))
    if min_importance > 0:
        mask &= stats["importance"] >= min_importance
    return mask
