"""
prefetch_weights.py — download the LPIPS backbone weights (torchvision VGG16 + AlexNet) once.

metrics.py fetches them lazily on first use, which fails on compute nodes without internet. Run
this on a login node, pointing TORCH_HOME at storage the compute nodes can read, and export the
same TORCH_HOME in the jobs:

    TORCH_HOME=/path/to/shared/torch_home python scripts/prefetch_weights.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import ImageMetrics  # noqa: E402

for net in ("vgg", "alex"):
    ImageMetrics(device="cpu", lpips_net=net)._lpips_model()
    print(f"lpips-{net}: ok")
print("weights cached under:", Path(torch.hub.get_dir()) / "checkpoints")
