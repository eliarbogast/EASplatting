# Results and figures

The single place for results (plans, decisions and open questions live in [`docs/EA_PLAN.md`](../docs/EA_PLAN.md)).
Every `python -m easplatting.experiment` run publishes its figure here automatically (`easplatting/plot_run.py`):
`runs/<run>.png` (the standard 4-panel plot) and `runs/<run>.json` (its summary), and it refreshes `compare.png`.
**Commit this folder with each run** (`results/` is git-ignored; the per-evaluation `log.jsonl` stays there).
Regenerate everything from `results/`: `python -m easplatting.plot_run --all`.

## Headline (2026-11-18; 4 genes, 3 seeds, 672 evals per GA/random run, sweeps 64 evals per gene)
% of Gaussians kept (lower is better); quality is always measured against the unpruned render.

**A. Fitness: mean PSNR >= 40 dB on 8 half-res train views** (`runs/<scene>_4g.png`)

| Scene | GA, mean (best) | Random | Importance sweep | Opacity sweep | Worst test view, dB: GA / imp. / opac. |
|---|---|---|---|---|---|
| bonsai | 45.4 (43.5) | 60.2 | **42.9** | 70.6 | 29 / 28 / 33 |
| counter | 45.7 (42.6) | 52.6 | **42.9** | 65.3 | 30 / 28 / 37 |
| garden | 55.2 (49.1) | 63.5 | **49.2** | 65.6 | 23 / 21 / 38 |
| bicycle | 52.3 (50.2) | 75.4 | **50.8** | 74.1 | 30 / 30 / 38 |

**B. Robust fitness: 32 views, mean >= 40 dB and worst view >= 35 dB** (`runs/<scene>_4g_v32_w35.png`)

| Scene | GA, mean (best) | Random | Importance sweep | Opacity sweep | Worst test view, dB: GA / imp. / opac. |
|---|---|---|---|---|---|
| bonsai | 61.7 (52.2) | 67.6 | **50.8** | 73.9 | 34 / 31 / 38 |
| counter | 54.2 (52.4) | 63.8 | **55.6** | 65.3 | 35 / 35 / 37 |
| garden | 58.6 (57.4) | 66.4 | **63.5** | 67.2 | 25 / 24 / 39 |
| bicycle | 66.2 (60.5) | 78.7 | **60.3** | 74.1 | 36 / 34 / 38 |

- **Importance gene is the big win:** 43-51% kept (A) vs 65-74% for the best opacity threshold, on all four scenes; its best cutoff
  is nearly the same everywhere (percentile 0.45-0.57). Max-scale and anisotropy alone prune almost nothing.
- **A hides tail damage.** The mean over 8 views looks fine (test mean 39-41 dB) but importance solutions have a worst
  test view of 21-30 dB (garden's mean test PSNR drops to 33.7). Diagnosis (garden): all-train mean 40.4 dB, worst train view 24 dB.
  More views alone (32, mean only) did not fix it; a worst-view floor did.
- **B is the honest number:** 8-14 points more kept than A; importance still beats opacity by 4-23 points. Floors do not fully transfer
  (garden's worst test view is ~25 dB).
- **GA vs 1-D sweep:** in A the GA never beats the importance sweep; in B it wins on counter and garden (1.4 / 4.9 points) and loses
  on bonsai and bicycle. The 64-point sweep overshoots the sharp feasible edge, so a finer sweep may erase those wins (EA_PLAN next step 2).
  The GA always beats random search at equal budget. GA means are noisy where the feasible region is thin (bonsai B: 52-74%).
- Importance = per-Gaussian blend weight over all train views (`render.gaussian_contributions`), cached in `results/cache/`; the gene removes
  the lowest-percentile Gaussians. Runs are deterministic (a rerun reproduced every number).

## Earlier result: 3 genes, 8 views (2026-11-13; scratch scripts, not reproducible from the repo)
% kept at 40 dB, GA / random / grid 9^3-12^3 / 1-D opacity sweep: bonsai 71.5 / 74.3 / 71.3-70.5 / **70.6**; counter 65.7 / 70.1 / 71.3-70.2 / **65.3**;
garden 66.4 / 72.6 / 75.0-69.9 / **65.6**; bicycle 74.2 / 77.7 / 83.1-77.4 / **74.1**. With three threshold genes a 64-eval opacity sweep
matches the GA (the best solutions switch scale and anisotropy off); the best opacity threshold varies ~4x across scenes (0.019-0.074).

## What the pruned scenes look like: `runs/<run>_renders.jpg`
One strip per run (`python -m easplatting.strip`, written automatically after each run; GPU). Columns: unpruned | best GA genome | best importance-sweep
genome | best opacity-sweep genome. Two held-out test views (the GA genome's **worst** view and the **median** one), each with three rows: the renders,
an error map (black = identical), and a 4x zoom of the window where the GA genome's error is highest (yellow box, same window in every column).
Example (`garden_4g_renders.jpg`): at 20.7 dB the importance/GA renders look the same as the reference at full size, but the zoom shows broken,
dithered detail along the roof edge and foliage where the opacity render (40.4 dB, 65.6% kept) stays clean: the 21-30 dB tail is real, local damage
in distant, low-importance regions.

## How to read `runs/<run>.png`
A best feasible % kept vs evaluations (GA/random per seed, sweeps dashed) | B every evaluation in the (% kept, mean PSNR) plane, constraint dashed |
C best genome per run | D the same genomes re-measured: fitness views, all train views, test views, worst test view (dashed lines: mean constraint,
worst-view floor). Colors: GA blue, random orange, importance sweep aqua, other sweeps gray. Run names: `<scene>_<genes>g[_v<views>][_w<floor>]`.

## Presentation kit
- **Story:** (1) prune pretrained 3DGS scenes with an evolved threshold genome, no retraining; (2) thresholds on opacity/scale/anisotropy
  barely beat a 1-D sweep (negative result); (3) an importance gene cuts the count by another ~25 points; (4) the mean-PSNR constraint hides
  badly damaged views, so we constrain the worst view too; (5) honest numbers: 51-66% kept at 40 dB mean / 35 dB worst; (6) next: Pareto front,
  evolved importance scores.
- **Slide-ready now:** `compare.png` (one-slide summary), any `runs/*.png` (method detail), `runs/*_renders.jpg` (what it looks like; the robust-setting
  strip per scene is the natural hero image), the tables above.
- **Worth making (not done):** a "what got removed" render (removed Gaussians only) and top-down kept/removed plot; threshold-sweep GIF (opacity 0 -> 0.2); importance and opacity histograms with the evolved cutoffs; per-view
  PSNR bars (the tail damage); FPS of pruned scenes (the speed claim is untested: we report counts, not FPS yet).
