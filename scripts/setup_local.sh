#!/usr/bin/env bash
# setup_local.sh — Python environment for the 3DGS pruning pipeline (Linux + NVIDIA GPU).
#
#   bash scripts/setup_local.sh                 # conda env "gsprune" (or .venv if no conda)
#   WITH_TRAINER=0 bash scripts/setup_local.sh  # skip gsplat's trainer (render/eval pretrained scenes only)
#
# Knobs (env vars): ENV_NAME, PY, TORCH, TORCHVISION, CU, GSPLAT, WITH_TRAINER
#
# What it installs
#   torch + torchvision (CUDA build)      -> download.pytorch.org
#   gsplat==1.5.3 (latest release)        -> prebuilt wheel if one exists for this torch/CUDA,
#                                            otherwise the PyPI wheel, which JIT-compiles CUDA
#                                            kernels on first use (needs nvcc; ~5-15 min once)
#   our deps: plyfile, pillow, torchmetrics[image], ninja
#   WITH_TRAINER=1: clones gsplat at the SAME tag (examples/ must match the library version)
#                   and installs examples/requirements.txt (fused-ssim etc. compile -> nvcc)
set -euo pipefail

ENV_NAME=${ENV_NAME:-gsprune}
PY=${PY:-3.10}
# Blackwell GPUs (compute capability >= 12.0, e.g. RTX 50xx) have no kernels in torch 2.4/cu124:
# they need torch >= 2.7 built with cu128. There are no prebuilt gsplat wheels for these, so gsplat
# JIT-compiles (nvcc >= 12.8 required; installed below via conda).
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ "${GPU_CC%%.*}" -ge 12 ] 2>/dev/null; then
  DEF_TORCH=2.8.0; DEF_TV=0.23.0; DEF_CU=cu128
else
  DEF_TORCH=2.4.1; DEF_TV=0.19.1; DEF_CU=cu124
fi
# Compile kernels only for this GPU (conda's activate scripts clobber $CC, hence GPU_CC).
[ -n "$GPU_CC" ] && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$GPU_CC}"
TORCH=${TORCH:-$DEF_TORCH}
TORCHVISION=${TORCHVISION:-$DEF_TV}
CU=${CU:-$DEF_CU}               # must be <= the CUDA version nvidia-smi reports
GSPLAT=${GSPLAT:-1.5.3}
WITH_TRAINER=${WITH_TRAINER:-1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "nvidia-smi failed: no NVIDIA driver?"; exit 1; }

echo "== Python env"
set +u   # conda's activate.d hooks (e.g. cuda-nvcc) reference unset variables
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda env list | grep -qE "^${ENV_NAME}\s" || conda create -y -n "$ENV_NAME" "python=$PY"
  conda activate "$ENV_NAME"
else
  python3 -m venv "$ROOT/.venv"
  source "$ROOT/.venv/bin/activate"
fi
python -m pip install --upgrade pip wheel

echo "== CUDA toolkit (nvcc) for JIT/extension builds"
if ! command -v nvcc >/dev/null 2>&1; then
  if [ -n "${CONDA_PREFIX:-}" ]; then
    CUV="${CU#cu}"; CUV="${CUV:0:2}.${CUV:2}"          # cu124 -> 12.4
    echo "nvcc not found; installing cuda-toolkit $CUV into the conda env"
    conda install -y -c "nvidia/label/cuda-${CUV}.0" cuda-toolkit || \
      echo "!! conda cuda-toolkit install failed — install a CUDA $CUV toolkit manually if builds fail"
    export CUDA_HOME="$CONDA_PREFIX"
  else
    echo "!! nvcc not found. Prebuilt gsplat wheel may still work; the trainer's extensions will not build."
  fi
else
  nvcc --version | tail -1
fi

# conda's cuda-toolkit keeps headers/libs under targets/x86_64-linux, which torch's extension
# builder (CUDA_HOME/include) doesn't search -> point the compilers there, and persist it in the
# env so every later shell (trainer, VACC-style reruns) builds the same way.
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/targets/x86_64-linux/include" ]; then
  T="$CONDA_PREFIX/targets/x86_64-linux"
  export CUDA_HOME="$CONDA_PREFIX" CPATH="$T/include${CPATH:+:$CPATH}" LIBRARY_PATH="$T/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  conda env config vars set CUDA_HOME="$CUDA_HOME" CPATH="$CPATH" LIBRARY_PATH="$LIBRARY_PATH" \
    ${TORCH_CUDA_ARCH_LIST:+TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"} >/dev/null
fi
set -u
echo "== PyTorch $TORCH ($CU)"
pip install "torch==$TORCH" "torchvision==$TORCHVISION" --index-url "https://download.pytorch.org/whl/$CU"

echo "== gsplat $GSPLAT"
TAG="pt$(echo "$TORCH" | cut -d. -f1,2 | tr -d .)$CU"      # e.g. pt24cu124
# A prebuilt wheel (version 1.5.3+pt24cu124) sorts above PyPI's plain 1.5.3, so pip takes it when
# it exists for this torch/CUDA; otherwise it falls back to the PyPI JIT build.
pip install "gsplat==$GSPLAT" --extra-index-url "https://docs.gsplat.studio/whl/$TAG"
if pip show gsplat | grep -q "^Version: .*+"; then
  echo "installed prebuilt gsplat wheel ($TAG)"
else
  echo "installed PyPI gsplat: CUDA kernels JIT-compile on first use (needs nvcc)"
fi

echo "== project deps"
pip install plyfile pillow "numpy<2" "torchmetrics[image]" ninja pytest

if [ "$WITH_TRAINER" = "1" ]; then
  echo "== gsplat trainer (examples @ v$GSPLAT)"
  mkdir -p "$ROOT/ext"
  [ -d "$ROOT/ext/gsplat" ] || git clone --depth 1 --branch "v$GSPLAT" https://github.com/nerfstudio-project/gsplat.git "$ROOT/ext/gsplat"
  # fused-ssim / fused-bilagrid import torch in setup.py -> must build against the installed torch
  # (and only for this GPU's architecture, which keeps the nvcc builds short).
  pip install setuptools wheel
  pip install --no-build-isolation -r "$ROOT/ext/gsplat/examples/requirements.txt"
fi

# Persistent kernel cache: compile once, reuse (also do this on the VACC).
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "== smoke test: compile/load gsplat kernels and render one Gaussian"
python - <<'EOF'
import time, torch, gsplat
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
print(f"gsplat {gsplat.__version__} on {torch.cuda.get_device_name()}: center pixel {[round(x, 3) for x in c]} "
      f"(expect ~[0.9, 0.45, 0.18]); first call took {time.time() - t:.1f}s")
assert c[0] > 0.5, "render looks wrong"
EOF

cat <<EOF

Done. In new shells run:
  conda activate $ENV_NAME      (or: source .venv/bin/activate)
  export TORCH_EXTENSIONS_DIR=$ROOT/.torch_extensions
EOF
