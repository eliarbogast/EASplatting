# EASplatting

Evolutionary search over post-hoc pruning thresholds for pretrained
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) scenes:
fewer Gaussians and faster rendering with minimal quality loss, no retraining. Rendering uses
[gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3; evaluation follows the
[NerfBaselines](https://nerfbaselines.github.io/mipnerf360) Mip-NeRF 360 protocol.
(EA class project: Eli + Areeb.)

## Status (2026-11-02)

- **Done:** the data -> splat -> render -> metrics pipeline runs on a real GPU. Bonsai matches
  NerfBaselines' published numbers (PSNR/SSIM/LPIPS-VGG) and a gsplat-trained truck matches the
  trainer's own eval. An interactive viewer lets you drag pruning thresholds and watch the result.
- **Next:** the EA itself (genome -> fitness -> random search -> GA). Open design questions are in
  [`docs/EA_PLAN.md`](docs/EA_PLAN.md).

## Quickstart (Linux + NVIDIA GPU, conda)

```bash
bash scripts/setup_local.sh && conda activate gsprune    # also handles RTX 50xx (Blackwell)
bash scripts/get_data.sh                                 # bonsai + checkpoint + Tanks&Temples (~4 GB)
python scripts/check_env.py --smoke                      # does this machine work? (paste the output if not)
python -m pytest -q tests                                # CPU tests with a fake renderer
python viewer.py --scene pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply \
    --cameras data/360_v2/bonsai --images images_2       # then open http://localhost:8080
```

Datasets, checkpoints and results are downloaded or generated locally and are git-ignored.

## Where to read what

| I want to... | Read |
|---|---|
| Run the pipeline, evaluate a scene, train a splat, use the viewer | [`docs/LOCAL_PIPELINE.md`](docs/LOCAL_PIPELINE.md) |
| Know what to build next, open questions, compute budget, plots | [`docs/EA_PLAN.md`](docs/EA_PLAN.md) |
| Prepare to run on the UVM VACC (GPUs, env, jobs, checklist) | [`docs/CLUSTER.md`](docs/CLUSTER.md) |
| See current status and the decisions already made | [`HANDOFF.md`](HANDOFF.md) |
| Look up background (metrics, fitness design, baselines, pitfalls) | [`docs/GS_PRUNING_NOTES.md`](docs/GS_PRUNING_NOTES.md) (long; reference) |
| See what the first real-GPU runs produced | [`docs/bring_up/`](docs/bring_up/README.md) |
| Rules for Claude Code sessions | [`CLAUDE.md`](CLAUDE.md) |

## Code map

| File | What |
|---|---|
| `gs_loader.py` | load / activate / prune / save scenes (Inria PLY, gsplat `.pt`) |
| `cameras.py` | COLMAP, Inria `cameras.json`, Blender -> `Cameras` (world->camera, OpenCV) |
| `render.py` | gsplat wrapper, baseline caching, contributions, FPS |
| `metrics.py` | the only PSNR / SSIM / LPIPS implementation |
| `evaluate.py` | scene + cameras -> renders -> metrics -> `report.json` (with cross-checks) |
| `viewer.py` | interactive threshold-pruning viewer (viser) |
| `export_gsplat_cameras.py` | cameras in gsplat-trainer's normalized frame |
| `scripts/` | `check_env.py` (will it work here?), env setup (also GPU-less cluster login nodes), data download (`download.py`, no wget/unzip needed), gsplat training, `bench_eval.py` (cost of one EA fitness evaluation on this GPU), `metrics_intuition.py` (what the metrics mean for pruning), kernel pre-warm, LPIPS weight prefetch |
| `tests/` | CPU tests with a fake gsplat |
