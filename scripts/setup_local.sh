#!/usr/bin/env bash
# setup_local.sh — Python environment for the 3DGS pruning pipeline (Linux + NVIDIA GPU).
#
#   bash scripts/setup_local.sh                 # conda env "gsprune" (or .venv if no conda)
#   WITH_TRAINER=0 bash scripts/setup_local.sh  # skip gsplat's trainer (render/eval pretrained scenes only)
#
# Knobs (env vars): ENV_NAME, PY, TORCH, TORCHVISION, CU, GSPLAT, WITH_TRAINER,
#                   TORCH_CUDA_ARCH_LIST (e.g. "8.0;9.0;12.0"), NO_GPU_OK=1, PRINT_CONFIG=1
#
# Torch stack by hardware: torch 2.8/cu128 by default (sm_70..sm_120, driver CUDA >= 12.8); torch
# 2.4.1/cu124 for drivers 12.4-12.7 and for pre-Volta GPUs (sm < 7.0, e.g. GTX 10xx); drivers older
# than CUDA 12.4 stop with a message (override CU/TORCH/TORCHVISION at your own risk).
# Check a machine first with: python scripts/check_env.py
#
# On a cluster login node (no GPU): NO_GPU_OK=1 TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" bash scripts/setup_local.sh
# builds the env without a GPU; then run scripts/smoke_gsplat.py once on a GPU node to compile
# gsplat's kernels into the shared cache. See docs/CLUSTER.md.
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

# Preflight: the few non-Python tools this script itself relies on.
missing=""
for t in sed sort tr head grep; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
[ "${WITH_TRAINER:-1}" = 1 ] && { command -v git >/dev/null 2>&1 || missing="$missing git"; }
if ! command -v conda >/dev/null 2>&1; then
  command -v python3 >/dev/null 2>&1 || missing="$missing python3(or-conda)"
fi
if [ -n "$missing" ]; then
  echo "!! missing required tools:$missing -- install them (or Miniforge for conda, no root needed) and re-run"; exit 1
fi

ENV_NAME=${ENV_NAME:-gsprune}
PY=${PY:-3.10}
# torch 2.8 + cu128 is the default: its wheels carry kernels for sm_70 through sm_120 (Volta ... Hopper
# ... Blackwell, incl. RTX 50xx), so one env serves a mixed-GPU cluster. The old torch 2.4/cu124 stack
# (kernels sm_50..sm_90) is used when the driver is older than CUDA 12.8, or the GPU is older than
# Volta. There are no prebuilt gsplat wheels for cu128, so gsplat JIT-compiles on first use (nvcc,
# installed below via conda).
ver_lt() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]; }
HAVE_GPU=0; GPU_CC=""; DRV_CUDA=""
if nvidia-smi >/dev/null 2>&1; then
  HAVE_GPU=1
  GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
  DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
  case "$GPU_CC" in [0-9]*.[0-9]*) ;; *) GPU_CC="" ;; esac   # old drivers can't report compute_cap
fi
OLD_GPU=0
if [ -n "$GPU_CC" ] && [ "${GPU_CC%%.*}" -lt 7 ]; then OLD_GPU=1; fi
if [ "$HAVE_GPU" = 1 ] && [ -n "$DRV_CUDA" ] && ver_lt "$DRV_CUDA" 12.4 && [ -z "${CU:-}" ]; then
  echo "!! The NVIDIA driver supports CUDA $DRV_CUDA only; the supported torch builds (cu124/cu128) need >= 12.4."
  echo "   Update the driver, or at your own risk: CU=cu118 TORCH=2.4.1 TORCHVISION=0.19.1 bash scripts/setup_local.sh"
  exit 1
elif [ "$OLD_GPU" = 1 ] || { [ "$HAVE_GPU" = 1 ] && [ -n "$DRV_CUDA" ] && ver_lt "$DRV_CUDA" 12.8; }; then
  DEF_TORCH=2.4.1; DEF_TV=0.19.1; DEF_CU=cu124
  [ "$OLD_GPU" = 1 ] && echo "note: GPU is sm_${GPU_CC/./} (older than Volta): using torch 2.4.1/cu124, the newest stack with its kernels"
else
  DEF_TORCH=2.8.0; DEF_TV=0.23.0; DEF_CU=cu128
fi
if [ "$HAVE_GPU" = 1 ] && [ -z "$GPU_CC" ]; then
  echo "note: could not read the GPU's compute capability (old driver?); set TORCH_CUDA_ARCH_LIST if kernel builds fail"
fi
# Which GPU architectures to compile for. Only an EXPLICIT TORCH_CUDA_ARCH_LIST is persisted in the
# conda env; the auto-detected one is used for this script's builds only (a login node or a
# different GPU type later must not inherit it). conda's activate scripts clobber $CC, hence GPU_CC.
ARCH_FROM_USER=${TORCH_CUDA_ARCH_LIST:+1}
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  if [ -n "$GPU_CC" ]; then
    export TORCH_CUDA_ARCH_LIST="$GPU_CC"
  elif [ "${NO_GPU_OK:-0}" = 1 ]; then
    echo "!! No GPU here and TORCH_CUDA_ARCH_LIST is unset: set it (e.g. VACC: \"8.0;9.0;12.0\")."; exit 1
  fi
fi
TORCH=${TORCH:-$DEF_TORCH}
TORCHVISION=${TORCHVISION:-$DEF_TV}
CU=${CU:-$DEF_CU}               # must be <= the CUDA version nvidia-smi reports
GSPLAT=${GSPLAT:-1.5.3}
WITH_TRAINER=${WITH_TRAINER:-1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "== GPU"
if [ "$HAVE_GPU" = 1 ]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
elif [ "${NO_GPU_OK:-0}" = 1 ]; then
  echo "no GPU on this host (NO_GPU_OK=1): building the env only; run scripts/smoke_gsplat.py on a GPU node later"
else
  echo "nvidia-smi failed: no NVIDIA driver? (on a login node without a GPU, use NO_GPU_OK=1 and set TORCH_CUDA_ARCH_LIST)"; exit 1
fi
echo "config: torch $TORCH ($CU), arch list '${TORCH_CUDA_ARCH_LIST:-}', gsplat $GSPLAT, trainer=$WITH_TRAINER"
[ "${PRINT_CONFIG:-0}" = 1 ] && exit 0

echo "== Python env"
set +u   # conda's activate.d hooks (e.g. cuda-nvcc) reference unset variables
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda env list | grep -qE "^${ENV_NAME}\s" || conda create -y -n "$ENV_NAME" "python=$PY"
  conda activate "$ENV_NAME"
else
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || { echo "!! python3 >= 3.9 is needed for the venv fallback (torch 2.8); install conda/Miniforge instead"; exit 1; }
  python3 -m venv "$ROOT/.venv" || { echo "!! python3 -m venv failed (on Debian/Ubuntu: apt install python3-venv); or install conda/Miniforge"; exit 1; }
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
    ${ARCH_FROM_USER:+TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"} >/dev/null
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

# Persistent kernel cache: compile once, reuse (on a cluster, point this at shared storage).
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

if [ "$HAVE_GPU" = 1 ]; then
  echo "== smoke test: compile/load gsplat kernels and render one Gaussian"
  python "$ROOT/scripts/smoke_gsplat.py"
else
  echo "== smoke test skipped (no GPU): run  python scripts/smoke_gsplat.py  on a GPU node to pre-warm the kernel cache"
fi

cat <<EOF

Done. In new shells run:
  conda activate $ENV_NAME      (or: source .venv/bin/activate)
  export TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
EOF
