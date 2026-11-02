"""
gs_loader.py — load / prune / save pretrained 3D Gaussian Splatting scenes.

Reads either:
  * a 3DGS-style binary .ply (Inria original repo, most viewers/exports), or
  * a gsplat simple_trainer checkpoint (.pt with ckpt["splats"])

and returns ONE canonical dict of raw (pre-activation) tensors, using gsplat's names:

  means      (N, 3)   positions
  quats      (N, 4)   rotation quaternion (w, x, y, z), UNnormalized
  scales     (N, 3)   LOG scales           -> exp() to get real sizes
  opacities  (N,)     LOGIT opacity        -> sigmoid() to get [0, 1]
  sh0        (N, 1, 3) degree-0 SH (base color)
  shN        (N, K, 3) higher-order SH (K = 15 for degree 3, 0 if none)

Use `activate()` before thresholding or rendering, `prune()` to apply a mask,
and `save_ply()` to write a pruned scene (for viewers / file-size numbers).

Requires: numpy, torch, plyfile   (pip install plyfile)

Quick look at a scene from the command line:
    python gs_loader.py path/to/point_cloud.ply
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814  # degree-0 SH constant: rgb = 0.5 + SH_C0 * sh0


# ---------------------------------------------------------------- loading

def load_ply(path, device="cpu"):
    """Load a 3DGS binary .ply into the canonical raw-tensor dict."""
    v = PlyData.read(str(path))["vertex"]
    names = [p.name for p in v.properties]

    def stack(cols):
        return np.stack([np.asarray(v[c], dtype=np.float32) for c in cols], axis=1)

    means = stack(["x", "y", "z"])
    sh0 = stack(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]          # (N, 1, 3)
    opacities = np.asarray(v["opacity"], dtype=np.float32)            # (N,)

    # Sort numerically: f_rest_10 must come after f_rest_9, not after f_rest_1.
    def by_index(prefix):
        return sorted((n for n in names if n.startswith(prefix)),
                      key=lambda n: int(n.split("_")[-1]))

    scales = stack(by_index("scale_"))                                 # (N, 3)
    quats = stack(by_index("rot_"))                                    # (N, 4)

    rest = by_index("f_rest_")
    n = means.shape[0]
    if rest:
        # Stored channel-major: [R coeffs..., G coeffs..., B coeffs...]
        K = len(rest) // 3
        shN = stack(rest).reshape(n, 3, K).transpose(0, 2, 1)         # (N, K, 3)
    else:
        shN = np.zeros((n, 0, 3), dtype=np.float32)

    raw = dict(means=means, quats=quats, scales=scales,
               opacities=opacities, sh0=sh0, shN=np.ascontiguousarray(shN))
    return {k: torch.from_numpy(a).to(device) for k, a in raw.items()}


def load_gsplat_ckpt(path, device="cpu"):
    """Load a gsplat simple_trainer checkpoint (.pt) into the canonical dict."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    s = ckpt["splats"] if "splats" in ckpt else ckpt
    out = {k: s[k].detach().to(device).float()
           for k in ("means", "quats", "scales", "opacities", "sh0", "shN")}
    out["opacities"] = out["opacities"].reshape(-1)
    return out


def load_scene(path, device="cpu"):
    """Dispatch on file extension."""
    path = Path(path)
    if path.suffix == ".ply":
        return load_ply(path, device)
    if path.suffix in (".pt", ".pth", ".ckpt"):
        return load_gsplat_ckpt(path, device)
    raise ValueError(f"Unknown scene format: {path.suffix}")


# ---------------------------------------------------------- activations

def activate(raw):
    """Return render-ready / threshold-ready values (does not modify `raw`)."""
    sh = torch.cat([raw["sh0"], raw["shN"]], dim=1)                    # (N, K+1, 3)
    return dict(
        means=raw["means"],
        quats=torch.nn.functional.normalize(raw["quats"], dim=-1),
        scales=torch.exp(raw["scales"]),                               # world units
        opacities=torch.sigmoid(raw["opacities"]),                     # [0, 1]
        colors=sh,                                                     # SH coeffs
        sh_degree=int(math.isqrt(sh.shape[1])) - 1,                    # 0..3
    )


def base_rgb(raw):
    """View-independent RGB color per Gaussian (handy for debugging/plots)."""
    return (0.5 + SH_C0 * raw["sh0"][:, 0, :]).clamp(0, 1)


# ------------------------------------------------------- pruning / saving

def prune(raw, keep_mask):
    """Keep only Gaussians where keep_mask is True. Works on raw or per-Gaussian dicts."""
    return {k: (v[keep_mask] if torch.is_tensor(v) and v.ndim > 0 else v)
            for k, v in raw.items()}


def save_ply(raw, path):
    """Write the canonical raw dict back out as a standard 3DGS .ply."""
    to_np = lambda t: t.detach().cpu().float().numpy()
    means, sh0, shN = to_np(raw["means"]), to_np(raw["sh0"]), to_np(raw["shN"])
    n, K = means.shape[0], shN.shape[1]

    f_rest = shN.transpose(0, 2, 1).reshape(n, 3 * K)                  # back to channel-major
    cols = ([("x", means[:, 0]), ("y", means[:, 1]), ("z", means[:, 2])]
            + [(c, np.zeros(n, np.float32)) for c in ("nx", "ny", "nz")]
            + [(f"f_dc_{i}", sh0[:, 0, i]) for i in range(3)]
            + [(f"f_rest_{i}", f_rest[:, i]) for i in range(3 * K)]
            + [("opacity", to_np(raw["opacities"]))]
            + [(f"scale_{i}", to_np(raw["scales"])[:, i]) for i in range(3)]
            + [(f"rot_{i}", to_np(raw["quats"])[:, i]) for i in range(4)])

    arr = np.empty(n, dtype=[(name, "f4") for name, _ in cols])
    for name, data in cols:
        arr[name] = data
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def bytes_per_gaussian(raw):
    """Uncompressed PLY bytes per Gaussian (62 floats * 4 = 248 for SH degree 3)."""
    return 4 * (3 + 3 + 3 + 3 * raw["shN"].shape[1] + 1 + 3 + 4)


# ---------------------------------------------------------------- summary

def summarize(raw):
    """Print distributions useful for picking threshold ranges for the EA genome."""
    a = activate(raw)
    n = raw["means"].shape[0]
    q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    pct = lambda x: "  ".join(f"{v:.3g}" for v in torch.quantile(x.float().cpu(), q).tolist())

    max_scale = a["scales"].max(dim=1).values
    volume = a["scales"].prod(dim=1)
    print(f"Gaussians:        {n:,}")
    print(f"SH degree:        {a['sh_degree']}")
    print(f"Approx. size:     {n * bytes_per_gaussian(raw) / 1e6:.1f} MB "
          f"({bytes_per_gaussian(raw)} B/Gaussian)")
    print("Percentiles:      1%  5%  25%  50%  75%  95%  99%")
    print(f"  opacity         {pct(a['opacities'])}")
    print(f"  max scale       {pct(max_scale)}")
    print(f"  log10 volume    {pct(torch.log10(volume + 1e-30))}")
    for t in (0.005, 0.01, 0.05):
        print(f"  opacity < {t:<6} {(a['opacities'] < t).float().mean().item():.1%} of Gaussians")


# Rendering with gsplat (for reference; needs a CUDA GPU):
#
#   from gsplat import rasterization
#   a = activate(prune(raw, mask))           # raw already on "cuda"
#   img, alpha, info = rasterization(
#       a["means"], a["quats"], a["scales"], a["opacities"], a["colors"],
#       viewmats,            # (C, 4, 4) world-to-camera, OpenCV convention
#       Ks,                  # (C, 3, 3) intrinsics
#       width, height,
#       sh_degree=a["sh_degree"],
#       backgrounds=bg,      # (C, 3) — must match the baseline renders!
#   )
#   img = img.clamp(0, 1)    # (C, H, W, 3), clamp before PSNR/SSIM


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python gs_loader.py <scene.ply | ckpt.pt>")
    summarize(load_scene(sys.argv[1]))
