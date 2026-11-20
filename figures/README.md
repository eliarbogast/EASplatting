# Results and figures

The single place for results (plans, decisions and open questions live in [`docs/EA_PLAN.md`](../docs/EA_PLAN.md)).
Every `python -m easplatting.experiment` run publishes its figure here automatically (`easplatting/plot_run.py`):
`runs/<run>/` (one folder per run: `plots.png` the standard 4-panel plot, `renders.jpg` the render overview, `summary.json`) and refreshes `compare.png`.
**Commit this folder with each run** (`results/` is git-ignored; the per-evaluation `log.jsonl` stays there).
Regenerate everything from `results/`: `python -m easplatting.plot_run --all`.

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

## Exploration: search spaces a 1-D sweep cannot cover (2026-11-19, branch `explore/harder-search-2026-11-19`)
EXPLORATORY: bonsai and counter, **2 seeds**, 1,312 evaluations per run, the main-run fitness (32 views, mean >= 40 dB, worst >= 35 dB, `max` importance), no significance tests yet.
The genome is a weight per per-Gaussian feature rank plus one importance cutoff per region (`easplatting/explore.py`); with weights fixed to `max` and one region it is
exactly the refined 1-D sweep, so every row nests the baseline. Warm = GA / CMA-ES start at the sweep optimum. % kept, mean of 2 seeds (lower is better):

| Search space (genes) | bonsai GA | bonsai CMA-ES | counter GA | counter CMA-ES |
|---|---|---|---|---|
| 1-D importance sweep (refined) | 51.0 | 51.0 | 42.7 | 42.7 |
| 5 feature weights, cold (6) | 46.6 | 43.5 | 47.4 | 41.3 |
| 5 feature weights, warm (6) | 44.0 | 42.6 | 42.2 | 41.0 |
| 4 depth shells, cold (4) / warm (4) | 43.0 / 42.8 | 42.4 / 42.3 | 43.0 / 42.7 | 42.5 / 42.4 |
| 8 spatial cells, warm (8) | 44.6 | 42.9 | 42.7 | 42.6 |
| **10 features, warm (11)** | 41.1 | 40.3 | 38.0 | 36.8 |
| 10 features, cold (11) | 43.3 | 39.8 | 40.8 | 37.6 |
| 10 features, random search (11) | 47.8 | | 47.8 | |
| 10 + repaired `learned2`, `fisherj` (13) | 41.0 | 38.8 | 38.0 | 36.4 |
| 10 features + 4 depth shells (14) | 41.6 | **37.8** | 40.1 | 36.5 |

The 10 features: `max`, `opacity`, `volume`, `views`, `camdist`, `sum`, `density`, `lg`, `fisher`, `learned` (percentile ranks; `explore.py`).
- **A richer space is where the search methods earn their place:** 6-13 points under the 1-D sweep on bonsai, 5-6 on counter. From a cold start random search is
  5-11 points worse than the GA and CMA-ES; CMA-ES beats the GA by 1-3 points. A warm start is not needed (cold CMA-ES lands within ~1 point of warm).
- **Depth shells and spatial cells only help on bonsai** (counter: no gain over the sweep). Shell cutoffs do not transfer to other scenes (below).
- **Matched size on the test views** (`explore_fronts.png`): at ~42% kept the evolved genomes are -0.8 dB vs the photos on bonsai against about -2.2 (`max`), -4.1 (`lg`)
  and -1.2 (`sum`) (interpolated between the 40% and 50% points); on counter -0.45 vs -0.6 (`max`). The worst test view only ties `max` on bonsai; on counter the 10-feature genome at 37% kept is ~2.5 dB better.
  `lg` (LightGaussian-style score, **no fine-tuning, our re-implementation**) is the weakest score at every size (-4.5 dB at 40% kept, worst view ~15 dB).
- **The floors do not hold on held-out views:** evolved genomes sit at test mean 39.6-40.0 dB and worst view 31-34 dB (floor 40 / 35) because the searchers stop at the constraint
  boundary; random search lands further from it and passes more often (bonsai 5-weight space: 2 of 2 runs, evolved: 0 of 4). A +1 dB margin on tau and worst-tau recovers the counter floor
  (worst test view 36.2 dB) for ~1.3 points kept, but not bonsai (worst test view still 32 dB).
  Diagnosis (`explore_eval gapcheck`, mean dB vs unpruned): on **bonsai** the gap is held-out poses (train -> test at the same resolution: evolved -0.3 mean / -1.6 worst,
  sweep -0.1 / 0); on **counter** it is resolution (half -> full res costs the evolved genome 1.85 dB, the sweep 0.8 dB; test and train views agree at half res). Fitness views vs all train
  views agree to 0.1 dB. Fitness on 8 views overfits badly (bonsai worst test view 20 dB); 32 and 64 views are similar.
- **Transfer (genome evolved on X, applied to Y, one scalar re-calibrated on Y's fitness views):** only the 10-feature genome carries over: vs Y's own sweep -9.5 (bonsai), -0.5 (counter), -4.2 (garden),
  -0.9 (bicycle) points kept; shell genomes +0.5..+12 (no gain); 5-weight genomes mixed. Some of the gain costs held-out quality (garden test 34.8 / 25.9 dB vs the sweep's 36.2 / 27.1).
- **Novel viewpoints** (`explore_eval stress`, cameras jittered up to one camera spacing and 10 degrees): no extra damage for evolved genomes; on counter every importance-based mask loses ~5 dB on its
  worst view at the largest jitter (`max` included), a weakness of train-view importance in general.
- **`learned` / `fisher` repair:** the gate learner's per-Gaussian gradients (~1e-10) were below Adam's eps (1e-8), so nothing moved; with the loss rescaled the gates separate
  (1-D sweep 100% -> 80.8% kept on bonsai, still far behind `max` at 51.0%). A Jacobian-based Fisher (`scores.fisher_jacobian`, render sensitivity, not photo loss) gives 91.3% kept. Both have good *mean*
  quality at 50-60% kept but worst views of 15-20 dB, which is why the worst-view floor rejects them. As extra features they add 0.4-1.5 points (CMA-ES).
- **New feature groups** (`features.py`: tail contribution, edge strength, pixel footprint, depth, neighbour density, scale / anisotropy, leave-fitness-views-out importance) added to the 10: bonsai CMA-ES -0.4 to -1.7 points
  (footprint and neighbour density best), counter -0.5 to +0.1, all 19 together -1.3 / +0.6, GA unchanged. Within seed noise (~1 point): the 10-feature set looks saturated.
- **FPS** (test views, full res, RTX 5070 Ti, `explore_eval fps`): unpruned -> evolved masks at 37-42% kept: bonsai 447 -> 568-631 fps (1.27-1.41x), counter 306 -> 437-493 (1.43-1.61x), garden / bicycle at
  50% kept (importance cutoff) 1.56x / 1.52x. **Gaussian count is a poor proxy for speed:** opacity pruning at the same count renders much faster (bonsai 1.63-1.73x, counter 1.75-1.96x), because it removes
  large faint Gaussians that importance keeps.
- **Negative results:** a fitness cache saves nothing (<2% duplicate genomes at 2 decimals); a 4-view multi-fidelity prefilter skips only 6-19% of evaluations (warm starts sit near the boundary), 0-20% time saved, no quality change.
- Feature-weight analysis (`scripts/explore_weights.py`): `max` carries ~0.6-0.9 of the normalized weight; the features are collinear (`sum` vs `max` Spearman 0.94), so weight vectors are not unique (cosine 0.68 between seeds).
- **10 seeds, +1 dB margin** (2026-11-20; 10 features, cold start, floors mean >= 41 dB / worst >= 36 dB on 32 fitness views, `seeds10_margin*`, Mann-Whitney one-sided):

  | scene (sweep kept) | method | % kept | test mean dB | test worst dB (min) |
  |---|---|---|---|---|
  | bonsai (52.3) | CMA-ES | 41.4 +- 1.0 | 40.4 | 32.0 (31.5) |
  | | GA | 44.6 +- 2.0 | 41.6 | 32.8 (31.7) |
  | | random | 52.6 +- 2.4 | 45.1 | 37.6 (32.1) |
  | counter (44.3) | CMA-ES | 38.7 +- 0.2 | 39.7 | 35.5 (34.5) |
  | | GA | 43.1 +- 1.0 | 40.8 | 36.1 (33.4) |
  | | random | 51.1 +- 3.2 | 43.3 | 37.8 (34.0) |

  CMA-ES < GA (p = 0.0003 bonsai, 0.0001 counter) and GA < random (p = 0.0001, 0.0006). Random keeps about as many as the sweep or more, so its higher test PSNR is just size. The margin does **not** close the
  held-out gap: CMA-ES test mean is below 41 dB on both scenes and bonsai's worst view stays ~32 dB (held-out poses, which a margin on fitness views cannot fix).
Reproduce: `python -m easplatting.explore combo --scene bonsai --features max,opacity,volume,views,camdist,sum,density,lg,fisher,learned --learn-weights --warm --methods ga,cmaes`, then
`python scripts/summarize_explore.py`; held-out evaluation `python -m easplatting.explore_eval transfer|fronts|stress|gapcheck|fps`; the figure `python scripts/plot_explore_fronts.py`. Raw rows: `results/explore/*.jsonl` (git-ignored).

## Comparability of runs (read before comparing tables)
- **Baseline tables A and B above** use the summed blend-weight importance, un-refined 64-point sweeps and no CMA-ES.
- **Importance-score screening** (`*_4g_v32_w35_i<mode>`, 1-D sweeps only) keeps everything of B the same (32 views, mean >= 40 dB, worst >= 35 dB, same
  scenes, cameras, fitness, 64-point sweeps) and changes only the score: `max`, `density`, `lg` (LightGaussian-style baseline). Directly comparable with B's
  importance-sweep column (the unchanged opacity/scale/anisotropy sweeps reproduce B's numbers exactly, which is the check that nothing else drifted).
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
- **Worth making (not done):** a "what got removed" render (removed Gaussians only) and top-down kept/removed plot; threshold-sweep GIF (opacity 0 -> 0.2); importance and opacity histograms with the evolved cutoffs; per-view
  PSNR bars (the tail damage); FPS of pruned scenes (the speed claim is untested: we report counts, not FPS yet).
