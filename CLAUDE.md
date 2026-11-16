# EASplatting

Evolving post-hoc pruning thresholds for pretrained 3D Gaussian Splatting scenes
(EA class project, Eli Arbogast + Areeb). A genome of thresholds → keep-mask over
Gaussians → render with gsplat → compare to unpruned renders → fitness. No retraining.

**Current phase: implementing the EA** (genome -> fitness -> random search -> GA). The local gsplat
bring-up is done and verified on Eli's machine (see `docs/bring_up/README.md`).

Start at `README.md` (status, quickstart, layout, doc map). `docs/EA_PLAN.md`: decisions, next steps, open
questions, compute budget, how to measure fitness. `docs/RUNNING.md`: how to run, check and measure things, and
the VACC checklist. `docs/BACKGROUND.md`: long reference (metrics, fitness options, baselines, pitfalls).

## Layout
Code lives in the `easplatting/` package; run tools as `python -m easplatting.<name>` from the repo root.
- `easplatting/gs_loader.py` load/activate/prune/save scenes (Inria PLY, gsplat `.pt`)
- `easplatting/cameras.py` COLMAP / Inria `cameras.json` / Blender -> `Cameras` (world->camera, OpenCV)
- `easplatting/render.py` gsplat wrapper, baseline caching, per-Gaussian contributions, FPS
- `easplatting/metrics.py` the ONLY PSNR/SSIM/LPIPS implementation
- `easplatting/thresholds.py` the genes as threshold -> keep-mask logic (`gene_stats`, `slider_bounds`,
  `keep_mask`); the genome decoder reuses it
- `easplatting/genome.py` (genome in [0,1]^3 -> thresholds, with "off" zones), `fitness.py` (`Individual`,
  feasibility-first `make_fitness_function`), `ea.py` (`evolutionary_algorithm`: mutation +
  tournament selection, elitism, no crossover; keep its comments short), `run_ea.py` (`python -m easplatting.run_ea`)
- `easplatting/evaluate.py` end-to-end check: scene + cameras -> renders -> metrics -> `report.json`
- `easplatting/viewer.py` interactive viser viewer for threshold pruning
- `easplatting/export_gsplat_cameras.py` cameras in gsplat-trainer's normalized frame
- `scripts/` `setup_local.sh` (`NO_GPU_OK=1` for GPU-less cluster login nodes), `get_data.sh`,
  `train_gsplat.sh`; `check_env.py` (machine report; ask for its output when debugging someone's setup;
  `--smoke-only` pre-warms gsplat, `--prefetch-weights` fetches LPIPS weights); `download.py` (replaces
  wget/curl/unzip; also range-extracts from remote zips); `measure.py` (`cost` feeds EA_PLAN's budget,
  `metrics` feeds its fitness assessment)
- `tests/` CPU tests with a fake gsplat (`tests/fake_gsplat`)
- `docs/archive/` history only (bring-up results); don't update it

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
- A gene's "off" (no pruning from it) must be exactly reachable: thresholds 0 / inf, as the viewer's
  slider ends do. `slider_bounds` stops the anisotropy slider at the 99.9th percentile, which is NOT off
  (it removes ~0.1% extreme Gaussians and caps quality at ~40 dB vs unpruned). Genome decoders need an
  explicit "off" zone.
- Same background color for baseline and pruned renders (black for real scenes).
- Pin gsplat **1.5.3** (latest release); `ext/gsplat` examples must be the same tag.

## Working here
- Tests: `python -m pytest -q tests` (CPU, no GPU needed). Keep them passing; add a test
  with the fake rasterizer for new plumbing.
- Real rendering needs Linux + NVIDIA. Cloud sessions can't run gsplat: say so rather than
  claiming GPU results.
- `data/`, `pretrained/`, `results/`, `ext/`, `.torch_extensions/` are git-ignored; never
  commit datasets, scenes or renders.
