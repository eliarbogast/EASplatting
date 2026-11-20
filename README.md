# EASplatting

Evolutionary search over post-hoc pruning thresholds for pretrained
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) scenes:
fewer Gaussians and faster rendering with minimal quality loss, no retraining. Rendering uses
[gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3; evaluation follows the
[NerfBaselines](https://nerfbaselines.github.io/mipnerf360) Mip-NeRF 360 protocol.
(EA class project: Eli + Areeb.)

## Status (2026-11-20)

- **Built:** data -> splat -> render -> metrics (matches NerfBaselines and gsplat's own eval), an interactive viewer, and the EA: a 4-gene genome
  (opacity, max scale, max anisotropy, importance percentile) with feasibility-first fitness, searched by GA, random, CMA-ES, NSGA-II, MAP-Elites and 1-D sweeps on
  bonsai, counter, garden and bicycle.
- **Main result** (full-res fitness and scores, mean PSNR >= 40 dB and worst view >= 35 dB vs the unpruned render, % of Gaussians kept by CMA-ES):
  bonsai 49.9, counter 44.0, garden 57.8, bicycle 55.7, against 64-73% for the best opacity-only threshold. The importance gene is the big win.
- **Search:** GA beats random search, but CMA-ES and a refined 1-D importance sweep match or beat it: the 4-gene space has one basin.
- **Lesson:** half-resolution fitness views and importance scores over-report quality on the outdoor scenes; use full res (`--fit-scale 1 --score-scale 1`).
- **Next:** a search space a sweep cannot cover (evolved importance score, per-region thresholds), bonsai / counter at full-res scores, FPS-aware objective,
  fine-tuned finals, talk visuals. Details, the plan and open questions: [`docs/EA_PLAN.md`](docs/EA_PLAN.md); all tables: [`figures/README.md`](figures/README.md).

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

Tests also run in GitHub Actions on every push. Every EA tool and its options: [`docs/RUNNING.md`](docs/RUNNING.md). Datasets, checkpoints and `results/`
are local and git-ignored; `figures/` is tracked.

## Where to read what

| I want to... | Read |
|---|---|
| Run, check or measure anything; use the viewer; run GPU jobs from Claude Code | [`docs/RUNNING.md`](docs/RUNNING.md) |
| Plan, decisions, open questions, compute budget, how fitness is measured | [`docs/EA_PLAN.md`](docs/EA_PLAN.md) |
| Results, figures, presentation ideas | [`figures/README.md`](figures/README.md) |
| Background (metrics, fitness options, baselines, pitfalls) | [`docs/BACKGROUND.md`](docs/BACKGROUND.md) (long) |
| History (first GPU results, VACC notes, early plans) | [`docs/archive/`](docs/archive/README.md) |
| Module-by-module layout and conventions (also the rules for Claude Code sessions) | [`CLAUDE.md`](CLAUDE.md) |

## Layout

```
easplatting/   the code; run a tool with  python -m easplatting.<name>
               loading and rendering: gs_loader, cameras, render, metrics, thresholds
               the EA: genome, fitness, ea, cmaes, nsga2, map_elites, scores, experiment (one scene, all searchers), front, rescore (full-res re-score)
               outputs: plot_run, plot_front, strip (render strips), evaluate (end-to-end check), viewer (viser)
scripts/       setup_local.sh, get_data.sh, train_gsplat.sh, check_env.py, download.py, measure.py
tests/         CPU tests with a fake gsplat
docs/          RUNNING.md, EA_PLAN.md, BACKGROUND.md, archive/ (history only)
figures/       TRACKED results: README.md, compare.png, runs/<run>/ (plots.png, renders.jpg, summary.json)
```

The weighted-feature genome, transfer and FPS experiments live on the unmerged branch `explore/harder-search-2026-11-19`.
