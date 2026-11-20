# Results and figures

The single place for results (plans, decisions and open questions live in [`docs/EA_PLAN.md`](../docs/EA_PLAN.md)).
Every `python -m easplatting.experiment` run publishes its figure here automatically (`easplatting/plot_run.py`):
`runs/<run>/` (one folder per run: `plots.png` the standard 4-panel plot, `renders.jpg` the render overview, `summary.json`) and refreshes `compare.png`.
**Commit this folder with each run** (`results/` is git-ignored; the per-evaluation `log.jsonl` stays there).
Regenerate everything from `results/`: `python -m easplatting.plot_run --all`.

> **Read the "Full-resolution fitness" section first:** everything from the headline to the fronts evolves on half-res fitness views, which over-reports quality on the outdoor scenes (garden, bicycle).

## Headline (2026-11-18; 4 genes, 3 seeds, 672 evals per GA/random run, sweeps 64 evals per gene)
% of Gaussians kept (lower is better); quality is always measured against the unpruned render.

**A. Fitness: mean PSNR >= 40 dB on 8 half-res train views** (`runs/<scene>_4g/`)

| Scene | GA, mean (best) | Random | Importance sweep | Opacity sweep | Worst test view, dB: GA / imp. / opac. |
|---|---|---|---|---|---|
| bonsai | 45.4 (43.5) | 60.2 | **42.9** | 70.6 | 29 / 28 / 33 |
| counter | 45.7 (42.6) | 52.6 | **42.9** | 65.3 | 30 / 28 / 37 |
| garden | 55.2 (49.1) | 63.5 | **49.2** | 65.6 | 23 / 21 / 38 |
| bicycle | 52.3 (50.2) | 75.4 | **50.8** | 74.1 | 30 / 30 / 38 |

**B. Robust fitness: 32 views, mean >= 40 dB and worst view >= 35 dB** (`runs/<scene>_4g_v32_w35/`)

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
  on bonsai and bicycle. The 64-point sweep overshoots the sharp feasible edge, so a finer sweep may erase those wins (it did: see the main runs below).
  The GA always beats random search at equal budget. GA means are noisy where the feasible region is thin (bonsai B: 52-74%).
- Importance = per-Gaussian blend weight over all train views (`render.gaussian_contributions`), cached in `results/cache/`; the gene removes
  the lowest-percentile Gaussians. Runs are deterministic (a rerun reproduced every number).

## Main runs, fronts and score comparison (2026-11-18; `max` importance, 32 views, mean >= 40 dB, worst >= 35 dB)
% kept (lower is better), GA/random 3 seeds x 672 evals, sweeps refined (12 bisection steps). NOT like-for-like with B (see below).
`runs/<scene>_4g_v32_w35_imax/`

| Scene | GA | Random | CMA-ES | Importance sweep (refined) | Opacity sweep |
|---|---|---|---|---|---|
| bonsai | 54.4 | 67.8 | **51.0** | **51.0** | 73.1 |
| counter | 44.6 | 55.4 | **42.7** | **42.7** | 64.1 |
| garden | 56.5 | 64.2 | **52.1** | **52.1** | 65.8 |
| bicycle | 54.7 | 75.8 | **52.3** | 52.2 | 73.1 |

- **GA earns nothing over the simplest searcher here:** it beats random by 9-21 points but loses to CMA-ES and the refined 1-D sweep on every scene
  (2-4 points), and those two agree, i.e. one basin that a single importance cutoff reaches. The earlier GA wins in B were sweep-resolution artifacts.
- **Score screening (1-D sweeps, mean % kept):** `max` 50.0 < `density` 61.8 < `lg` 65.9 (`sum` 57.6). Max-over-views is the best of the cheap scores.
- **Variants** (`_cov`, `_L0.5`) move results by 1-2 points (garden at 0.5 dB loss: ~47%); view choice is not the bottleneck.
- **`learned` gates and `fisher` scores (`*_ilearned`, `*_ifisher`) did not work in this simple form:** the importance sweep is 100% kept (learned) or 82-93% (fisher),
  worse than the opacity sweep, and GA/CMA-ES fall back to opacity (64-76%). Treat as a failed baseline, not evidence about the published scores;
  likely causes: gate logits barely separate (300 steps, 48 views) and squared-gradient scores are dominated by a few heavy-tailed Gaussians. Not debugged.
- **Fronts** (`runs/<scene>_front_v32_imax/`, hypervolume, higher is better; 3 seeds):

| Scene | NSGA-II | MAP-Elites | Random | 1-D sweep |
|---|---|---|---|---|
| bonsai | 0.408 | 0.464 | 0.335 | **0.490** |
| counter | 0.496 | 0.389 | 0.396 | **0.535** |
| garden | 0.416 | 0.389 | 0.324 | **0.461** |
| bicycle | 0.407 | 0.272 | 0.304 | **0.450** |

  The importance sweep alone has the best front on all scenes; NSGA-II beats random everywhere, MAP-Elites only on bonsai/garden. MAP-Elites archives fill
  90-96% of cells, so the descriptor space is easy to cover but the extra diversity does not buy quality.

## Full-resolution fitness (2026-11-20; `max` importance, 32 views, mean >= 40 dB, worst >= 35 dB; `runs/<scene>_4g_v32_w35_imax_res1/`)
**Why:** every table above evolves on HALF-resolution fitness views. Re-scoring the half-res runs' best genomes at full resolution (`python -m easplatting.rescore`, no new search) shows the floors
do not hold there: bonsai holds, counter misses (CMA-ES / importance sweep mean 39.3 dB, worst 35.2; 0/3 CMA-ES seeds hold both floors), garden misses by a lot (mean 36.7, worst 28.9; 0/3 GA, 0/3 CMA-ES),
bicycle likewise (34.8 / 26.9). The opacity sweep holds on every scene. **Fix:** evolve on full-res fitness views (`experiment.py --fit-scale 1`), 10 seeds per searcher, refined sweeps (12 bisection steps),
importance scores first unchanged (computed once at half res, cached), then also at full res (`--score-scale 1`, last section below). The half-res runs above stay as the ablation.

% kept (lower is better), mean over seeds; "half" = tables above (3 seeds, half-res fitness), "full" = full-res fitness (10 seeds):

| Scene | GA half / full | Random half / full | CMA-ES half / full | Importance sweep half / full | Opacity sweep half / full |
|---|---|---|---|---|---|
| bonsai | 54.4 / 53.1 | 67.8 / 65.2 | 51.0 / **49.9** | 51.0 / **49.9** | 73.1 / 73.1 |
| counter | 44.6 / 46.6 | 55.4 / 57.3 | 42.7 / **44.0** | 42.7 / **44.0** | 64.1 / 63.8 |
| garden | 56.5 / 65.1 | 64.2 / 75.4 | 52.1 / **63.1** | 52.1 / 68.5 | 65.8 / 64.8 |
| bicycle | 54.7 / 67.4 | 75.8 / 78.3 | 52.3 / **65.0** | 52.2 / 65.3 | 73.1 / 71.4 |

Held-out check of the full-res runs (test views, full res, CMA-ES: mean over seeds of the mean / minimum over seeds of the worst view; floors 40 / 35): bonsai 39.9 / 38.2, counter 39.9 / 35.4,
garden 40.8 / 32.5, bicycle 45.0 / 38.6. Every run holds both floors on the full-res fitness views (`rescore` on `*_res1`: 10/10 per searcher on every scene).

- **The half-res error is a size effect on the outdoor scenes.** Indoor scenes barely move (bonsai -1.1, counter +1.3 points kept for the importance sweep / CMA-ES). Garden and bicycle move +11 / +13 points: their half-res
  numbers were not honest, because small Gaussians that half-res rendering hides matter at full res.
- **Importance still beats opacity, by much less outdoors:** bonsai 49.9 vs 73.1 (-23), counter 44.0 vs 63.8 (-20), bicycle 65.0 vs 71.4 (-6), garden 63.1 vs 64.8 (-2, a tie: GA 65.1 and the refined importance sweep 68.5 are
  worse than opacity). The "importance gene is the big win" claim holds on bonsai and counter only.
- **Searchers (10 seeds, one-sided Mann-Whitney):** GA < random on every scene (p <= 0.0014); CMA-ES < GA (p = 0.0001 bonsai, 0.004 counter, 0.005 garden, 0.02 bicycle). CMA-ES ties the refined importance sweep on bonsai, counter,
  bicycle (differences 0.0, 0.0, -0.3 points) and beats it on garden by 5.4 points, but the garden sweep is not trustworthy there: its feasible edge is not monotone (68.5% kept at mean 45 dB, far above the floor). GA loses to the
  sweep by 2-3 points on bonsai, counter and bicycle. So the claim stays "evolutionary search beats random, CMA-ES matches the sweep", not "the GA wins".
- **A margin on fitness views does not close garden's held-out gap:** test worst view 32.5 dB (CMA-ES, minimum over seeds; 33.2 on average) vs the opacity sweep's 38. That gap is held-out poses (see the exploration section on bonsai), not resolution.

### Full-res importance scores (2026-11-20; `runs/<scene>_4g_v32_w35_imax_res1_sres1/`, `--fit-scale 1 --score-scale 1`)
The importance scores were still computed from half-res renders. Computing them at full res too (10 seeds, same floors) recovers most of the outdoor loss. % kept, mean over seeds ("fit-res" = full-res fitness, half-res scores, from the table above):

| Scene | GA fit-res / +scores | Random | CMA-ES | Importance sweep | Opacity sweep (unchanged) |
|---|---|---|---|---|---|
| garden | 65.1 / **59.6** | 75.4 / 69.1 | 63.1 / **57.8** | 68.5 / **57.1** | 64.8 |
| bicycle | 67.4 / **57.8** | 78.3 / 68.8 | 65.0 / **55.7** | 65.3 / **55.6** | 71.4 |

- **The half-res scores were the cause** of most of the outdoor loss: -5 to -11 points kept, so the importance gene again beats opacity (garden 57.8 vs 64.8, bicycle 55.7 vs 71.4: -7 and -16). Indoor scenes were not re-run (they barely moved at full-res fitness).
- Every run holds both floors at full res (`rescore`: 10/10 per searcher). Held-out test, CMA-ES (mean over seeds of the mean / minimum over seeds of the worst view): garden 39.8 / 35.9, bicycle 39.5 / 36.7. Garden's held-out worst view rose from 32.5 to 35.9 dB, so the gap above was partly a score-resolution effect.
- Searchers: GA < random (p = 0.0007 garden, 0.007 bicycle) and CMA-ES < GA (p = 0.002, 0.0001). CMA-ES ties the refined sweep (57.8 vs 57.1, 55.7 vs 55.6); the garden sweep is now sane (feasible, 57.1 %).
- **Use the `_res1_sres1` runs as the main results for garden and bicycle.** `_res1` stays as the ablation.
- **Not done:** importance scores are still computed from half-res renders (a likely cause of the outdoor over-pruning; recomputing at full res is the next test); fitness at 0.75 resolution or full-res worst-view only (counter, exploration branch:
  0.75 meets the worst floor, misses the mean by 0.3 dB).

## Comparability of runs (read before comparing tables)
- **Baseline tables A and B above** use the summed blend-weight importance, un-refined 64-point sweeps and no CMA-ES.
- **Importance-score screening** (`*_4g_v32_w35_i<mode>`, 1-D sweeps only) keeps everything of B the same (32 views, mean >= 40 dB, worst >= 35 dB, same
  scenes, cameras, fitness, 64-point sweeps) and changes only the score: `max`, `density`, `lg` (LightGaussian-style baseline). Directly comparable with B's
  importance-sweep column (the unchanged opacity/scale/anisotropy sweeps reproduce B's numbers exactly, which is the check that nothing else drifted).
- **Full-res scores** (`*_imax_res1_sres1`, garden and bicycle only) additionally compute the importance scores at full res; otherwise identical to `*_res1`.
- **Full-res main runs** (`*_imax_res1`) change only the fitness view resolution (full instead of half) and use 10 seeds instead of 3; same scenes, views, floors, scores, sweeps. Held-out diagnostics there use the fitness resolution on 48 spread train views plus full-res test views.
- **Main runs with the chosen score** (`--methods ga,random,cmaes,sweep --sweep-refine 12`) are NOT like-for-like with B: they add CMA-ES, refine the sweeps
  (12 bisection steps, so sweep numbers drop a little), and use the new score, all at once. We did not rerun B's old score with the refined sweeps and CMA-ES
  (decided 2026-11-18) so those three changes cannot be separated.
- **Variant runs** (`_cov` = fitness views chosen by camera coverage, `_L0.5` = tau derived from an allowed real loss of 0.5 dB vs the photos) change one
  fitness setting each, with the main runs' searchers.
- **Fronts** (`*_front_*`) are a different experiment: no constraint, quality = worst fitness-view PSNR, judged by hypervolume (`python -m easplatting.front`; figure `runs/<scene>_front_*/front.png`), so
  their numbers cannot be compared with the constrained % kept tables except by reading a front at a given quality.

## Earlier result: 3 genes, 8 views (2026-11-13; scratch scripts, not reproducible from the repo)
% kept at 40 dB, GA / random / grid 9^3-12^3 / 1-D opacity sweep: bonsai 71.5 / 74.3 / 71.3-70.5 / **70.6**; counter 65.7 / 70.1 / 71.3-70.2 / **65.3**;
garden 66.4 / 72.6 / 75.0-69.9 / **65.6**; bicycle 74.2 / 77.7 / 83.1-77.4 / **74.1**. With three threshold genes a 64-eval opacity sweep
matches the GA (the best solutions switch scale and anisotropy off); the best opacity threshold varies ~4x across scenes (0.019-0.074).

## What the pruned scenes look like: `runs/<run>/renders.jpg` and `runs/<run>/images/`
`python -m easplatting.strip` (written automatically after each run; GPU). **`renders.jpg`** is the overview: columns unpruned | best GA genome |
best importance-sweep genome | best opacity-sweep genome; two held-out test views (the GA genome's **worst** view and the **median** one), each with
three rows: the renders, an error map (black = identical), and a zoom (shown 4x larger) of the window where the GA genome's error is highest (yellow box, same
window in every column). The overview is a downscaled JPEG.

**`images/`** holds every picture of that overview on its own, lossless and at full resolution (8-bit PNG, not resized), with a descriptive name
`<worst|median>_view_<test image>__<method>__<kept>pct_kept__<psnr>dB[__error|__zoom4x].png` (`zoom4x` = the yellow-box window at native pixels, i.e. 1/4 of the image's width and height), and an `images/README.md` index (captions + thumbnails,
renders on GitHub) with `index.json` (the same, machine readable, incl. the zoom box). Full-resolution images are ~17 MB per run, so they always
exist in `results/<run>/images/` but are committed only when asked for (`--publish-images`, or `python -m easplatting.plot_run results/<run> --images`).
**Committed now:** the four robust-setting runs (`*_4g_v32_w35`). Regenerate any run's images without rerunning the search: `python -m easplatting.strip results/<run>`.

Example (`runs/garden_4g_v32_w35/`): see the strip. Earlier mean-only example (`garden_4g`): at 20.7 dB the importance/GA renders look the same as the
reference at full size, but the zoom shows broken, dithered detail along the roof edge and foliage where the opacity render (40.4 dB, 65.6% kept)
stays clean: the 21-30 dB tail is real, local damage in distant, low-importance regions.

## How to read `runs/<run>/plots.png`
A best feasible % kept vs evaluations (GA/random per seed, sweeps dashed) | B every evaluation in the (% kept, mean PSNR) plane, constraint dashed |
C best genome per run | D the same genomes re-measured: fitness views, all train views, test views, worst test view (dashed lines: mean constraint,
worst-view floor). Colors: GA blue, random orange, importance sweep aqua, other sweeps gray. Run names: `<scene>_<genes>g[_v<views>][_w<floor>]`.

## How to read `runs/<scene>_front_*/front.png`
A size-quality fronts on the fitness views, every seed (higher and further left wins) | B hypervolume of the front found so far vs evaluations |
C the best front of each method re-measured on the held-out test views | D the MAP-Elites archive (best quality per cell of kept % and importance share).
Colors: NSGA-II blue, MAP-Elites aqua, random orange, sweeps gray.

## Presentation kit
- **Story:** (1) prune pretrained 3DGS scenes with an evolved threshold genome, no retraining; (2) thresholds on opacity/scale/anisotropy
  barely beat a 1-D sweep (negative result); (3) an importance gene cuts the count by another ~25 points; (4) the mean-PSNR constraint hides
  badly damaged views, so we constrain the worst view too; (5) honest numbers: 51-66% kept at 40 dB mean / 35 dB worst; (6) negative result on the
  algorithm: the GA beats random search but not CMA-ES or a refined importance sweep (one basin), and the Pareto-front searchers (NSGA-II, MAP-Elites)
  do not beat the sweep's own front; (7) next: a search space a sweep cannot cover (evolved importance score, per-region thresholds).
- **Slide-ready now:** `compare.png` (one-slide summary), any `runs/*/plots.png` (method detail), `runs/*/renders.jpg` and `images/` (what it looks like; the robust-setting
  strip per scene is the natural hero image), the tables above (note the comparability caveats).
- **Worth making (not done), best first:**
  1. Side-by-side orbit video, unpruned vs best genome on the same camera path, with a "kept 48%, 40.0 dB" overlay (needs GPU; build on bonsai / counter first, add garden / bicycle once settled).
  2. "What got removed": removed-only render and error x5 next to the final render, plus a top-down kept/removed plot (the `viewer.py` modes already render these).
  3. Wipe slider or steps 100 -> 70 -> 50 -> 40 % kept; threshold-sweep GIF (opacity 0 -> 0.2).
  4. Evolution animation: best mask at generations 0, 5, 10, ... of one run (needs per-generation best genomes; check the logs, else re-run).
  5. Method race (random / GA / CMA-ES / sweep) with the final render under each; the 10-seed data supports it.
  6. Half-res vs full-res failure: one view that passes at half res and shows a hole or floater at full res.
  7. Cross-scene summary table with thumbnails, kept %, mean / worst PSNR, FPS.
  8. Hosted web splat viewer (e.g. SuperSplat) on exported pruned `.ply`, plus the file-size saving as a number.
  9. Importance and opacity histograms with the evolved cutoffs; per-view PSNR bars (tail damage).
  Label every figure with resolution and floors. Flow: problem (scene size) -> side-by-side video -> method -> evolution -> race and table -> half-res lesson -> limits.
