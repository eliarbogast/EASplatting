"""
metrics.py — the ONE implementation of PSNR / SSIM / LPIPS for this project.

Use it everywhere (EA fitness, baselines, final tables) so numbers are comparable.
Built on torchmetrics, the same backend gsplat's simple_trainer uses for eval,
so results on the same views/images should match its stats/val_step*.json.

Conventions
  * Inputs: (V, H, W, 3) float images in [0, 1] (gsplat render layout).
  * Each metric is computed PER VIEW, then averaged — how 3DGS papers and
    gsplat report them (mean of per-image PSNR, not PSNR of the mean MSE).
  * SSIM: torchmetrics default (11x11 Gaussian window, sigma 1.5, data_range 1).
    The Inria repo's own ssim() differs in the 3rd-4th decimal; don't mix them.
  * LPIPS:
      "vgg"  (default) net_type="vgg", normalize=True ([0,1] inputs rescaled to
             [-1,1] as LPIPS expects) — reproduces NerfBaselines' `lpips_vgg`
             (bonsai: 0.262 here vs 0.254 published; the raw variant gives 0.207).
      "alex" net_type="alex", normalize=True — gsplat simple_trainer's default and
             NerfBaselines' `lpips`. Use when cross-checking gsplat's own eval stats.
      "vgg-raw" net_type="vgg", normalize=False — the Inria 3DGS repo convention
             (feeds [0,1] images unscaled; numbers come out LOWER). Only for
             comparing against paper tables that used it; never mix with "vgg".
    First use downloads backbone weights (torchvision), so needs internet once.

    m = ImageMetrics(device="cuda", lpips_net="vgg")
    per_view = m(pred, gt)            # {"psnr": [...], "ssim": [...], "lpips": [...]}
    means    = summarize(per_view)    # {"psnr": 25.1, ...}

`psnr_fast` is a dependency-free per-view PSNR for the EA inner loop.
"""

import math

import torch


def psnr_fast(pred, gt, max_db=None):
    """Per-view PSNR (V,) in dB for (V, H, W, 3) images in [0, 1]. inf if identical unless capped."""
    mse = ((pred.float() - gt.float()) ** 2).flatten(1).mean(dim=1)
    db = -10.0 * torch.log10(mse)
    return db.clamp(max=max_db) if max_db is not None else db


class ImageMetrics:
    def __init__(self, device="cuda", lpips_net="vgg", psnr_cap=None):
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,
            structural_similarity_index_measure,
        )
        self._psnr = peak_signal_noise_ratio
        self._ssim = structural_similarity_index_measure
        self.device = device
        self.psnr_cap = psnr_cap
        self.lpips_net = lpips_net
        self._lpips = None                        # lazy: weights download on first use

    def _lpips_model(self):
        if self._lpips is None:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            if self.lpips_net == "vgg":
                m = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
            elif self.lpips_net == "vgg-raw":
                m = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False)
            elif self.lpips_net == "alex":
                m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
            else:
                raise ValueError(f"lpips_net must be 'vgg', 'vgg-raw' or 'alex', got {self.lpips_net!r}")
            self._lpips = m.to(self.device).eval()
        return self._lpips

    @torch.no_grad()
    def __call__(self, pred, gt, lpips=True):
        """pred, gt: (V, H, W, 3) in [0, 1]. Returns dict of per-view float lists."""
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}")
        p = pred.to(self.device, torch.float32).clamp(0, 1).permute(0, 3, 1, 2)   # (V, 3, H, W)
        g = gt.to(self.device, torch.float32).clamp(0, 1).permute(0, 3, 1, 2)
        out = {"psnr": [], "ssim": []}
        if lpips:
            out["lpips"] = []
            model = self._lpips_model()
        for i in range(p.shape[0]):
            pi, gi = p[i:i + 1], g[i:i + 1]
            v = self._psnr(pi, gi, data_range=1.0).item()
            if self.psnr_cap is not None:
                v = min(v, self.psnr_cap)
            out["psnr"].append(v)
            out["ssim"].append(self._ssim(pi, gi, data_range=1.0).item())
            if lpips:
                out["lpips"].append(model(pi, gi).item())
                model.reset()                      # don't accumulate state across calls
        return out


def summarize(per_view):
    """Mean and std of each metric list. inf PSNRs (identical images) are kept as inf in the mean."""
    res = {}
    for k, vals in per_view.items():
        if not vals:
            continue
        t = torch.tensor(vals, dtype=torch.float64)
        res[k] = float(t.mean())
        res[k + "_std"] = float(t.std()) if len(vals) > 1 else 0.0
    return res


def merge(a, b):
    """Concatenate two per-view dicts (for accumulating across render batches)."""
    for k, v in b.items():
        a.setdefault(k, []).extend(v)
    return a


if __name__ == "__main__":
    # Self-check: identical images, a known-MSE image, and a shifted image.
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ImageMetrics(device=dev)
    x = torch.rand(2, 64, 64, 3)
    y = (x + 0.01).clamp(0, 1)
    print("identical:", summarize(m(x, x, lpips=False)))
    r = m(x, y, lpips=False)
    exp = -10 * math.log10(((x - y) ** 2).mean().item())
    print(f"+0.01 offset: psnr {r['psnr'][0]:.2f} dB (expected ~{exp:.2f}), ssim {r['ssim'][0]:.4f}")
    print("psnr_fast:", psnr_fast(x, y).tolist())
