# EASplatting

Evolutionary search over post-hoc pruning thresholds for pretrained
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) scenes:
fewer Gaussians and faster rendering with minimal quality loss, no retraining. Rendering uses
[gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3; evaluation follows the
[NerfBaselines](https://nerfbaselines.github.io/mipnerf360) Mip-NeRF 360 protocol.
(EA class project: Eli + Areeb.)

## Status (2026-11-19)

- **Done:** the data -> splat -> render -> metrics pipeline runs on a real GPU and matches NerfBaselines' published numbers and gsplat's own eval;
  an interactive viewer; a measured evaluation cost (~10 ms per million Gaussians). The **EA is built**: genome -> feasibility-first fitness ->
  GA, random search, CMA-ES, NSGA-II, MAP-Elites and 1-D sweeps, run on four Mip-NeRF 360 scenes (bonsai, counter, garden, bicycle).
- **Findings** ([`figures/README.md`](figures/README.md)): an importance-percentile gene prunes far more than opacity alone (43-51% kept vs 65-74% at
  40 dB mean PSNR; 51-66% once the worst view is also constrained). But the GA only beats random search: CMA-ES and a refined 1-D importance sweep match
  or beat it, because the 4-gene space has a single basin.
- **Next:** a search space a sweep cannot cover (evolve the importance score itself, or per-region thresholds), fixing the failed learned/Fisher score
  baselines, FPS of pruned scenes, fine-tuning finals. Plan and open questions: [`docs/EA_PLAN.md`](docs/EA_PLAN.md).

## Quickstart (Linux + NVIDIA GPU)

```bash
bash scripts/setup_local.sh && conda activate gsprune    # also handles RTX 50xx (Blackwell)
bash scripts/get_data.sh                                 # bonsai + checkpoint + Tanks&Temples (~4 GB)
python scripts/check_env.py --smoke                      # does this machine work? (paste the output if not)
python -m pytest -q tests                                # CPU tests with a fake renderer
python -m easplatting.viewer --scene pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply \
    --cameras data/360_v2/bonsai --images images_2       # then open http://localhost:8080
python -m easplatting.experiment --scene bonsai --genes 4 --seeds 3   # GA vs random vs sweeps, a minute or two; publishes into figures/
```

The tests also run in GitHub Actions on every push (CPU torch, fake gsplat). Every EA tool and its options: [`docs/RUNNING.md`](docs/RUNNING.md) section 9.

Datasets, checkpoints and `results/` are downloaded or generated locally and are git-ignored; `figures/` is tracked.

## Where to read what

| I want to... | Read |
|---|---|
| Run, check or measure anything; use the viewer; run GPU jobs from Claude Code | [`docs/RUNNING.md`](docs/RUNNING.md) |
| Know the plan, decisions made, open questions, compute budget, how to measure fitness | [`docs/EA_PLAN.md`](docs/EA_PLAN.md) |
| See results and figures (one plot per run, a comparison across runs, presentation notes) | [`figures/README.md`](figures/README.md) |
| Look up background (metrics, fitness options, baselines, pitfalls) | [`docs/BACKGROUND.md`](docs/BACKGROUND.md) (long; reference) |
| See history: first real-GPU results, VACC notes, the first EA test, early plan questions | [`docs/archive/`](docs/archive/README.md) |
| Rules for Claude Code sessions | [`CLAUDE.md`](CLAUDE.md) |

## Layout

```
easplatting/           the code; run a tool with  python -m easplatting.<name>
  gs_loader.py           load / activate / prune / save scenes (Inria PLY, gsplat .pt)
  cameras.py             COLMAP, Inria cameras.json, Blender -> Cameras (world->camera, OpenCV)
  render.py              gsplat wrapper, baseline caching, contributions, FPS
  metrics.py             the only PSNR / SSIM / LPIPS implementation
  thresholds.py          pruning genes as threshold -> keep-mask logic
  genome.py, fitness.py  genome -> thresholds (4th gene = importance percentile); feasibility-first fitness (render fitness views, PSNR vs unpruned)
  scores.py              gradient-based importance scores (learned gates, Fisher); sum / max / density / lg are in experiment.py
  ea.py, experiment.py   minimal evolutionary algorithm (mutation + tournament selection); GA / random / CMA-ES / 1-D sweeps on one scene
  nsga2.py, map_elites.py, cmaes.py   NSGA-II (+ hypervolume), MAP-Elites, CMA-ES, in the style of ea.py
  front.py, plot_front.py   size-quality fronts and archives: NSGA-II / MAP-Elites vs random vs sweeps; their figure
  explore.py             harder search spaces: genome = weights over per-Gaussian feature ranks + per-region cutoffs (nests the 1-D sweep); FPS; cached extra scores
  explore_eval.py        held-out evaluation of those genomes: cross-scene transfer, fronts vs photos, novel-view stress, train/test x half/full-res gap check, FPS
  features.py            extra per-Gaussian features (tail contribution, edge strength, footprint, depth, neighbour density)
  plot_run.py            one standard PNG per run + a comparison PNG, published into figures/
  strip.py               render strip per run (unpruned vs best genomes, error maps, zoom); GPU
  evaluate.py            scene + cameras -> renders -> metrics -> report.json (with cross-checks)
  viewer.py              interactive threshold-pruning viewer (viser)
  export_gsplat_cameras.py   cameras in gsplat-trainer's normalized frame
scripts/               setup_local.sh, get_data.sh, train_gsplat.sh  (shell)
                       check_env.py (machine report, gsplat smoke test, LPIPS weights)
                       download.py (resumable download, unzip, range-extract from a remote zip)
                       measure.py (cost of a fitness evaluation; what the metrics mean)
                       summarize_explore.py, explore_weights.py, plot_explore_fronts.py (tables and figure for the explore.py runs)
tests/                 CPU tests with a fake gsplat (tests/fake_gsplat)
docs/                  RUNNING.md, EA_PLAN.md, BACKGROUND.md, archive/ (history only)
.github/workflows/    tests.yml: CPU tests on every push
figures/               TRACKED results: README.md (results summary), compare.png, runs/<run>/ with plots.png, renders.jpg, summary.json, optional images/ (written by experiment.py)
```
