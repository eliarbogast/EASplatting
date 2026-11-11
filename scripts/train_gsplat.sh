#!/usr/bin/env bash
# train_gsplat.sh — train a scene with gsplat's simple_trainer, then export its cameras.
#
#   bash scripts/train_gsplat.sh                                   # truck, 7k steps (smoke test, ~5-10 min)
#   STEPS=30000 bash scripts/train_gsplat.sh                       # full-length baseline
#   SCENE=data/360_v2/bonsai FACTOR=2 bash scripts/train_gsplat.sh # Mip-NeRF 360 indoor (outdoor: FACTOR=4)
#     (needs full-res images: FULLRES=1 bash scripts/get_data.sh mipnerf360 bonsai — gsplat re-downscales
#      JPEGs itself into images_<f>_png, so its eval GT differs slightly from the official images_<f>)
#
# Outputs in $OUT:
#   ckpts/ckpt_<STEPS-1>_rank0.pt   splats (raw log/logit params) — load with gs_loader
#   ply/point_cloud_<STEPS-1>.ply   same scene as PLY (gsplat drops NaN splats on export)
#   stats/val_step<STEPS-1>.json    gsplat's own test-view PSNR/SSIM/LPIPS(alex) — cross-check target
#   cameras.json (+ _meta.json)     cameras in the trainer's NORMALIZED world frame
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCENE=${SCENE:-data/tandt/truck}
FACTOR=${FACTOR:-1}
STEPS=${STEPS:-7000}
NAME=$(basename "$SCENE")
OUT=${OUT:-results/${NAME}_gsplat_$((STEPS / 1000))k}
LAST=$((STEPS - 1))
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$ROOT/.torch_extensions}

cd "$ROOT/ext/gsplat/examples"
python simple_trainer.py default \
  --data_dir "$ROOT/$SCENE" --data_factor "$FACTOR" --result_dir "$ROOT/$OUT" \
  --max_steps "$STEPS" --eval_steps "$STEPS" --save_steps "$STEPS" \
  --save_ply --ply_steps "$STEPS" \
  --disable_viewer --disable_video
cd "$ROOT"

python -m easplatting.export_gsplat_cameras --data "$SCENE" --factor "$FACTOR" \
  --examples ext/gsplat/examples --out "$OUT/cameras.json"

echo
echo "Next:"
echo "  python -m easplatting.evaluate --scene $OUT/ckpts/ckpt_${LAST}_rank0.pt --cameras $OUT/cameras.json \\"
echo "      --split test --lpips-net alex --gsplat-stats $OUT/stats/val_step${LAST}.json --out results/eval_${NAME}_gsplat"
