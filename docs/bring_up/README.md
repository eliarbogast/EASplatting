# Bring-up results (2026-11-02)

First real-GPU runs: RTX 5070 Ti (Blackwell, sm_120), torch 2.8.0+cu128, gsplat 1.5.3 (JIT-built,
first call 34 s), conda env `gsprune`. The files here are small copies of the git-ignored
`results/eval_*/` outputs (no renders). Commands: `docs/LOCAL_PIPELINE.md` steps 2-3.

| Run (files) | Result |
|---|---|
| **Bonsai**, NerfBaselines 3DGS checkpoint, 1,247,650 Gaussians, 37 test views at 1559x1039 (`bonsai_nb3dgs_*`) | PSNR 32.235 / SSIM 0.9421 / LPIPS-VGG 0.2622 vs NB 32.197 / 0.9412 / 0.2538 (diffs +0.038 dB / +0.0009 / +0.0084); all checks pass. Frustum 31.9% visible. Opacity > 0.005 keeps 87.9% at 60 dB vs unpruned. 13 ms/view batched, ~436 FPS single view, 4.6 GB peak |
| **Truck**, gsplat 7k steps, 2.05 M Gaussians, 2:16 training (`truck_gsplat7k_*`) | PSNR 23.829 / SSIM 0.8527 / LPIPS-alex 0.1340 vs the trainer's own 23.827 / 0.8526 / 0.1340 (`matches_gsplat`) |

The NerfBaselines checkpoint URL pattern, zip layout and metrics-json keys all held
(`point_cloud/iteration_30000/point_cloud.ply`; json `metrics.{psnr,ssim,lpips,lpips_vgg}`).

## Fixes made during bring-up

- `setup_local.sh`: Blackwell autodetect (torch 2.8/cu128, conda CUDA 12.8 toolkit, sm_120 JIT),
  `set -u` vs conda activate hooks, CUDA include paths persisted in the env, `--no-build-isolation`
  for fused-ssim / fused-bilagrid, pytest.
- `render.py`: gsplat 1.5.3 asserts on per-camera `backgrounds` when `packed=True`, so the
  background is composited by us (the fake rasterizer now rejects that case too).
- LPIPS-VGG: our old raw-[0,1] default gave 0.2071 on bonsai vs NerfBaselines' 0.2538. Same renders
  with `normalize=True` give 0.2622 (alex: 0.1328 vs their 0.1289), so NB rescales to [-1,1].
  `metrics.py` `vgg` now does that; `vgg-raw` keeps the Inria convention; `evaluate.crosscheck`
  enforces LPIPS (tol 0.01).
