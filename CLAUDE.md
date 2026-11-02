# EASplatting

Evolving post-hoc pruning thresholds for pretrained 3D Gaussian Splatting scenes
(EA class project, Eli Arbogast + Areeb). A genome of thresholds → keep-mask over
Gaussians → render with gsplat → compare to unpruned renders → fitness. No retraining.

**Current phase: local gsplat bring-up.** Don't implement genome/fitness/EA code until a real
gsplat run on Eli's Linux GPU machine passes `evaluate.py`'s checks (see `HANDOFF.md`).

Read `HANDOFF.md` for current status and next steps; `docs/GS_PRUNING_NOTES.md` for
background (metrics, fitness design, baselines, pitfalls); `docs/LOCAL_PIPELINE.md` to run it.

## Layout
- `gs_loader.py` load/activate/prune/save scenes (Inria PLY, gsplat `.pt`)
- `cameras.py` COLMAP / Inria `cameras.json` / Blender → `Cameras` (world→camera, OpenCV)
- `render.py` gsplat wrapper, baseline caching, per-Gaussian contributions, FPS
- `metrics.py` the ONLY PSNR/SSIM/LPIPS implementation
- `evaluate.py` end-to-end check: scene + cameras → renders → metrics → `report.json`
- `export_gsplat_cameras.py` cameras in gsplat-trainer's normalized frame
- `scripts/` env setup, data download, gsplat training, zip range-fetcher
- `tests/` CPU tests with a fake gsplat (`tests/fake_gsplat`)

## Conventions that must not drift
- Scene dicts hold RAW params: log scales, logit opacity, unnormalized quats, SH split
  `sh0 (N,1,3)` / `shN (N,K,3)`. Call `activate()` before thresholding or rendering;
  genes threshold ACTIVATED values (sigmoid opacity, exp scale).
- PLY `f_rest_*` is channel-major; never hand-roll SH reshapes — use gs_loader.
- Cameras: world→camera, OpenCV. Inria-trained / NerfBaselines-3DGS scenes are in the raw
  COLMAP frame. gsplat-`simple_trainer` scenes are in a NORMALIZED frame → use
  `export_gsplat_cameras.py`. `frustum_check` ~0% visible = frame mismatch.
- Split: name-sorted cameras, every 8th = test (llffhold 8). Evolve on train views only;
  report on test views.
- Benchmark protocol = NerfBaselines Mip-NeRF 360: official `images_4` (outdoor) /
  `images_2` (indoor), LPIPS-VGG with inputs rescaled to [-1,1]
  (`metrics.py` default `vgg`, `normalize=True`; verified against NB's published bonsai numbers).
  `--lpips-net alex` only when cross-checking gsplat's own eval; `vgg-raw` (Inria's unscaled
  convention, ~0.05 lower) only to compare with paper tables, never mixed with `vgg`.
- Same background color for baseline and pruned renders (black for real scenes).
- Pin gsplat **1.5.3** (latest release); `ext/gsplat` examples must be the same tag.

## Working here
- Tests: `python -m pytest -q tests` (CPU, no GPU needed). Keep them passing; add a test
  with the fake rasterizer for new plumbing.
- Real rendering needs Linux + NVIDIA. Cloud sessions can't run gsplat: say so rather than
  claiming GPU results.
- `data/`, `pretrained/`, `results/`, `ext/`, `.torch_extensions/` are git-ignored; never
  commit datasets, scenes or renders.
