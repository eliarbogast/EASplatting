# EASplatting

Evolutionary search over post-hoc pruning thresholds for pretrained
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) scenes:
fewer Gaussians and faster rendering with minimal quality loss, no retraining. Rendering uses
[gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3; evaluation follows the
[NerfBaselines](https://nerfbaselines.github.io/mipnerf360) Mip-NeRF 360 protocol.
(EA class project: Eli + Areeb.)

## Status (2026-11-11)

- **Done:** the data -> splat -> render -> metrics pipeline runs on a real GPU and matches NerfBaselines' published
  numbers and gsplat's own eval. An interactive viewer lets you drag pruning thresholds and watch the result.
  We measured what a fitness evaluation costs (~10 ms per million Gaussians) and what the metrics mean for pruning.
- **Next:** the EA itself (genome -> fitness -> random search -> GA). Open design questions: [`docs/EA_PLAN.md`](docs/EA_PLAN.md).

## Quickstart (Linux + NVIDIA GPU)

```bash
bash scripts/setup_local.sh && conda activate gsprune    # also handles RTX 50xx (Blackwell)
bash scripts/get_data.sh                                 # bonsai + checkpoint + Tanks&Temples (~4 GB)
python scripts/check_env.py --smoke                      # does this machine work? (paste the output if not)
python -m pytest -q tests                                # CPU tests with a fake renderer
python -m easplatting.viewer --scene pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply \
    --cameras data/360_v2/bonsai --images images_2       # then open http://localhost:8080
```

Datasets, checkpoints and results are downloaded or generated locally and are git-ignored.

## Where to read what

| I want to... | Read |
|---|---|
| Run, check or measure anything; use the viewer; run on the UVM VACC | [`docs/RUNNING.md`](docs/RUNNING.md) |
| Know the plan, decisions made, open questions, compute budget, how to measure fitness | [`docs/EA_PLAN.md`](docs/EA_PLAN.md) |
| See the first EA test run and how the low-res fitness views are rendered | [`docs/EA_TEST.md`](docs/EA_TEST.md) |
| Look up background (metrics, fitness options, baselines, pitfalls) | [`docs/BACKGROUND.md`](docs/BACKGROUND.md) (long; reference) |
| See the first real-GPU results | [`docs/archive/bring_up/`](docs/archive/bring_up/README.md) |
| Rules for Claude Code sessions | [`CLAUDE.md`](CLAUDE.md) |

## Layout

```
easplatting/           the code; run a tool with  python -m easplatting.<name>
  gs_loader.py           load / activate / prune / save scenes (Inria PLY, gsplat .pt)
  cameras.py             COLMAP, Inria cameras.json, Blender -> Cameras (world->camera, OpenCV)
  render.py              gsplat wrapper, baseline caching, contributions, FPS
  metrics.py             the only PSNR / SSIM / LPIPS implementation
  thresholds.py          pruning genes as threshold -> keep-mask logic
  genome.py, fitness.py  genome -> thresholds; feasibility-first fitness (render fitness views, PSNR vs unpruned)
  ea.py, run_ea.py       minimal evolutionary algorithm (mutation + tournament selection) and a test run on bonsai
  evaluate.py            scene + cameras -> renders -> metrics -> report.json (with cross-checks)
  viewer.py              interactive threshold-pruning viewer (viser)
  export_gsplat_cameras.py   cameras in gsplat-trainer's normalized frame
scripts/               setup_local.sh, get_data.sh, train_gsplat.sh  (shell)
                       check_env.py (machine report, gsplat smoke test, LPIPS weights)
                       download.py (resumable download, unzip, range-extract from a remote zip)
                       measure.py (cost of a fitness evaluation; what the metrics mean)
tests/                 CPU tests with a fake gsplat (tests/fake_gsplat)
docs/                  RUNNING.md, EA_PLAN.md, BACKGROUND.md, archive/
```
