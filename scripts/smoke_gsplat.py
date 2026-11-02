"""
smoke_gsplat.py — compile/load gsplat's CUDA kernels and render one Gaussian.

Run once on a GPU node (it is also the kernel-cache pre-warm job on a cluster):
    export TORCH_EXTENSIONS_DIR=<shared dir> TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"
    python scripts/smoke_gsplat.py
"""

import time

import torch
import gsplat
from gsplat import rasterization

t = time.time()
d = "cuda"
img, alpha, meta = rasterization(
    means=torch.tensor([[0., 0., 3.]], device=d), quats=torch.tensor([[1., 0, 0, 0]], device=d),
    scales=torch.full((1, 3), 0.3, device=d), opacities=torch.tensor([0.9], device=d),
    colors=torch.tensor([[1., 0.5, 0.2]], device=d),
    viewmats=torch.eye(4, device=d)[None], Ks=torch.tensor([[[100., 0, 64], [0, 100., 64], [0, 0, 1]]], device=d),
    width=128, height=128)
torch.cuda.synchronize()
c = img[0, 64, 64].tolist()
print(f"gsplat {gsplat.__version__} on {torch.cuda.get_device_name()} "
      f"(sm_{''.join(map(str, torch.cuda.get_device_capability()))}): center pixel {[round(x, 3) for x in c]} "
      f"(expect ~[0.9, 0.45, 0.18]); first call took {time.time() - t:.1f}s")
assert c[0] > 0.5, "render looks wrong"
