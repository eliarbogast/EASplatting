# Local pipeline: data → splat → renders → metrics

Goal: prove every stage works on a local Linux + NVIDIA machine before spending VACC time.
No fitness/EA yet. Commands run from the repo root.

## 0. Environment (once)

```bash
bash scripts/setup_local.sh            # conda env "gsprune": torch 2.4.1+cu124, gsplat 1.5.3, trainer deps
conda activate gsprune
export TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions
python -m pytest -q tests              # CPU plumbing tests (fake renderer) — should all pass
```

Blackwell GPUs (RTX 50xx, compute capability 12.x) are auto-detected: the script switches to
torch 2.8.0 + cu128, and conda's CUDA 12.8 toolkit (needed because there is no prebuilt gsplat
wheel for these, so gsplat JIT-compiles for sm_120). It also persists `CUDA_HOME`, `CPATH`,
`LIBRARY_PATH` and `TORCH_CUDA_ARCH_LIST` in the conda env. Needs conda (Miniforge is fine) and
no root. torch 2.4.1+cu124 has no sm_120 kernels, so don't override `CU` on these cards.

The script ends with a one-Gaussian render on the GPU. If gsplat came from PyPI (no
prebuilt wheel for your torch/CUDA), the first call JIT-compiles CUDA kernels (needs
`nvcc`, 5–15 min, cached afterwards). Override versions with env vars, e.g.
`CU=cu121 TORCH=2.4.1 bash scripts/setup_local.sh` (CU must be ≤ the CUDA version
`nvidia-smi` reports). `WITH_TRAINER=0` skips gsplat's trainer if you only evaluate
pretrained scenes.

## 1. Data

```bash
bash scripts/get_data.sh               # default set below (~3-4 GB total)
```

| What | Where | Why |
|---|---|---|
| Mip-NeRF 360 **bonsai** (images_2 + COLMAP) | `data/360_v2/bonsai` | main benchmark, small indoor scene |
| NerfBaselines 3DGS checkpoint for bonsai + its metrics json | `pretrained/nb/gaussian-splatting/mipnerf360/bonsai/` | baseline trained on train split only; published test metrics to cross-check |
| Inria Tanks&Temples + Deep Blending inputs | `data/tandt`, `data/db` | small COLMAP scenes for training tests |
| Inria pretrained **truck** | `pretrained/inria/truck/point_cloud.ply` | quick render test (may have seen test views) |

More scenes: `bash scripts/get_data.sh mipnerf360 garden counter` and
`bash scripts/get_data.sh nb-checkpoints garden counter`. Mip-NeRF 360 scenes are
range-extracted from the 12 GB zip (flowers/treehill come from the separate extra-scenes
zip), so you only download what you ask for. Add `FULLRES=1` if you'll train on Mip-NeRF
360 with gsplat (its parser re-downscales from full-res).

## 2. Render + metrics on a pretrained scene (main check)

```bash
NB=pretrained/nb/gaussian-splatting/mipnerf360/bonsai
python cameras.py data/360_v2/bonsai --images images_2 --scene $NB/point_cloud.ply   # frustum check
python evaluate.py --scene $NB/point_cloud.ply --cameras data/360_v2/bonsai --images images_2 \
    --split test --nb-metrics $NB/bonsai.json --prune-opacity 0.005 --fps --cache-baselines \
    --out results/eval_bonsai_nb3dgs
```

Pass criteria (all in `results/eval_bonsai_nb3dgs/report.json`):
- `checks.frustum_median_visible` well above 5% (tens of %).
- `checks.gt_size_matches_camera: true`.
- Test PSNR/SSIM within ~0.1 dB / 0.003 of NerfBaselines' numbers
  (`checks.matches_nerfbaselines`; tolerances 0.1 dB / 0.003 / 0.01). This is the strongest evidence the whole chain
  (loader, cameras, renderer, metrics, protocol) is right.
- `compare/*.png` look right: GT | render | error×5.
- `prune_demo`: kept fraction and PSNR vs unpruned (first taste of the trade-off).

## 3. Train a splat yourself (data → splat)

```bash
bash scripts/train_gsplat.sh                       # truck, 7k steps — smoke test
OUT=results/truck_gsplat_7k
python evaluate.py --scene $OUT/ckpts/ckpt_6999_rank0.pt --cameras $OUT/cameras.json \
    --split test --lpips-net alex --gsplat-stats $OUT/stats/val_step6999.json \
    --out results/eval_truck_gsplat_7k
```

`checks.matches_gsplat` should be true (our metrics vs the trainer's own eval on the same
views). Then `STEPS=30000` for a real baseline. Note the gsplat-trained scene lives in a
normalized world frame — always pair it with `$OUT/cameras.json`, never raw COLMAP.

## 4. Look at it: interactive viewer

```bash
python viewer.py --scene $NB/point_cloud.ply --cameras data/360_v2/bonsai --images images_2
# open http://localhost:8080   (over SSH: ssh -L 8080:localhost:8080 <host>)
```

Sliders threshold opacity, max scale and max anisotropy (log10; the end marked "off" keeps
everything). Live readout of kept count, estimated size, render time and the current view's PSNR vs
the unpruned render. View modes: pruned, unpruned, removed only (what the thresholds delete) and
error x5. Buttons save the thresholds (`results/viewer/thresholds.json`) or the pruned `.ply`.
`--cameras` adds a jump-to-camera dropdown. The opacity-0.005 preset reproduces `evaluate.py`'s
87.9% kept on bonsai.

## 5. Record the machine

Keep `report.json` from each machine (it records GPU, torch, gsplat versions and timings).
Compare `timing.render_s_per_view` and `fps_*` with the VACC later to size the EA budget:
evals/s ≈ 1 / (n_fitness_views × render_s_per_view).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| frustum check ~0% | gsplat-trained scene with raw COLMAP cameras → use exported `cameras.json`; or wrong scene/cameras pair |
| PSNR < 15 dB, image looks shifted/garbage | wrong `--images` folder (intrinsics vs image size), or distorted (non-PINHOLE) COLMAP |
| PSNR a bit low, images look fine | background mismatch (NeRF-Synthetic: white), or `--rasterize-mode` vs how it was trained |
| GT size warning | `--images` doesn't match the resolution the scene was evaluated at |
| first render takes minutes | gsplat JIT compile (once); keep `TORCH_EXTENSIONS_DIR` set |
| LPIPS off vs a paper | different LPIPS convention/net; we follow NerfBaselines (VGG, inputs rescaled to [-1,1]); Inria-repo/paper tables feed raw [0,1] and read ~0.05 lower (`--lpips-net vgg-raw`) |
| truck test PSNR ≫ 25 dB | Inria pretrained model saw the test views; use NB checkpoints for honest numbers |
