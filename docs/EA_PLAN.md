# EA plan

Status, decisions, next steps, the compute budget, how to measure fitness, and the open design questions, for
Eli + Areeb as we write the EA. "Lean" = a suggestion, not a decision. How to run things: [`RUNNING.md`](RUNNING.md).
Background and fitness-design options: [`BACKGROUND.md`](BACKGROUND.md) sections 5-8.

## Status and next steps (2026-11-17; findings: see "Findings so far and planned directions")

**Done and verified on a real GPU** (RTX 5070 Ti; results in [`RUNNING.md`](RUNNING.md) section 7): scene loading,
cameras, rendering, metrics (match NerfBaselines and gsplat's own eval), the evaluation report, the interactive
viewer, the threshold -> mask logic (`easplatting/thresholds.py`), data/setup/measurement scripts.
**EA pipeline on `main`:** genome decoding (3 thresholds + optional importance-percentile gene), feasibility-first
fitness (mean-PSNR floor, optional worst-view floor, `--views K`), a mutation + tournament GA, and
`python -m easplatting.experiment` (GA vs random vs 1-D sweeps; JSONL row per evaluation; all-train / test / worst-view
PSNR per best genome). 46 CPU tests. **Results:** the importance gene cuts kept % from 65-74 (opacity only) to 43-51 at
40 dB, but those solutions hide badly damaged views; a robust fitness gives an honest 51-66%. See the two result
sections below.

**Decisions already made**
- **Renderer: gsplat 1.5.3** (latest release; 1.6.0 on main is unreleased with a different trainer API). Batched
  multi-camera forward renders from Python; a mask is just tensor indexing.
- **Benchmark: NerfBaselines Mip-NeRF 360 protocol**: official `images_4` (outdoor) / `images_2` (indoor), llffhold-8
  test split (every 8th name-sorted camera), LPIPS-VGG on [-1,1]-rescaled inputs. Unpruned baseline = NerfBaselines'
  `gaussian-splatting` checkpoints (trained on the train split only, published per-scene metrics). Inria's own
  pretrained models may include test views, so use them only for quick tests.
- **Dev scene: bonsai** (indoor, 1.25 M Gaussians); outdoor scenes run to several million.
- **Tanks & Temples:** Inria's `tandt_db` (not NerfBaselines' T&T, which is preprocessed differently).
- **Fitness reference:** the unpruned renders (cached by `evaluate --cache-baselines`), not the photos; final tables
  report both. Evolve on train views, report on test views.

**Next steps** (in order; the other three "planned directions" below still stand)
1. **Pareto front** (NSGA-II over opacity / scale / importance; hypervolume vs the sweeps' fronts). A single constrained
   point is what a sweep does well; the front is where an EA can still win.
2. **Finer / adaptive 1-D sweeps as the baseline**: 64 points overshoot a sharp feasibility edge (fit PSNR 46-50 dB at
   the best sweep point), which flatters the GA on counter and garden. Re-check before claiming a GA win.
3. **Transfer the floor to held-out views**: garden's worst test view is still ~25 dB with a 32-view floor. Try views
   chosen for coverage (pose clustering) instead of evenly spaced, and an allowed-real-loss constraint per scene.
4. Baselines: LightGaussian-style percentile pruning at equal count, CMA-ES; optional fine-tuning of finals only.

**Other ideas (not started; one line each)**
- *Evolve the importance score:* genome = weights over per-Gaussian features (opacity, volume, view count, blend weight,
  distance to camera hull); a high-dimensional space a sweep cannot cover, which is where an EA should matter.
- *Spatial / per-region genes:* separate thresholds per depth shell or octree cell (background vs foreground).
- *Per-view-aware importance:* score each Gaussian by its worst-view contribution (max over views, not the sum), which
  targets the tail damage the mean hides.
- *SH-degree cap gene:* drop higher SH bands (size in bytes, not count); needs the byte-size objective (Q2).
- *Transfer:* does a genome (esp. the ~0.5 importance percentile) evolved on one scene work on another (Q13)?
- *Multi-fidelity:* score all genomes on 4 views, re-score the top quarter on 32; keeps the robust fitness cheap.
- *Quality-diversity (MAP-Elites)* over (kept %, worst-view PSNR) instead of one constrained optimum.
- *Importance on fitness views only*, or leave-views-out importance, to cut the train/test gap at its source.

## Findings so far and planned directions (2026-11-13)

**Four-scene comparison** (bonsai, counter, garden, bicycle; NerfBaselines checkpoints; each validated against
NerfBaselines' published metrics). Fitness as in `experiment.py`: minimise kept fraction s.t. PSNR >= 40 dB vs the
unpruned render on 8 half-res train views. GA = 32 parents + 64 children x 10 generations (672 evals), 3 seeds;
random search = same 672 evals; grids are deterministic; "1D opacity" = 64 evals with max-scale and anisotropy off.

| Scene | GA mean (best) | Random | Grid 9^3 | Grid 12^3 | **1D opacity (64 evals)** | Best opacity threshold |
|---|---|---|---|---|---|---|
| bonsai | 71.5 (70.5) | 74.3 | 71.3 | 70.5 | **70.6** | 0.049 |
| counter | 65.7 (64.2) | 70.1 | 71.3 | 70.2 | **65.3** | 0.074 |
| garden | 66.4 (65.5) | 72.6 | 75.0 | 69.9 | **65.6** | 0.035 |
| bicycle | 74.2 (74.1) | 77.7 | 83.1 | 77.4 | **74.1** | 0.019 |

(% of Gaussians kept; lower is better. Held-out test-view PSNR vs unpruned was 40.2-41.9 dB for the GA bests.)

What it says:
- **With these three genes, a 1D opacity sweep does what the GA does** (matches or beats the GA mean on all four
  scenes, with 64 evals instead of 672). The best solutions mostly turn max-scale and anisotropy off. The outdoor
  scenes did not change this. The one hint of a gain from the scale gene is counter (GA bests 64.2-64.8% vs 65.3%,
  about 1 point, 3 seeds), too small to call.
- **The GA still beats random search by 3-6 points on every scene**, but random search is a weak baseline here: it
  wastes slack (ends 2-8 dB above the floor on several seeds). The 3D grids lose to the 1D sweep because they spend
  their budget on genes that do not matter (9 opacity values only).
- **The real value found so far is per-scene calibration:** the best opacity threshold varies about 4x across scenes
  (0.019 to 0.074), so one global threshold is infeasible on some scenes and wasteful on others. A 64-point sweep
  gets it in a few seconds.
- **The ceiling (65-74% kept at 40 dB vs unpruned, roughly 1 dB real loss vs the photos on bonsai) comes from the gene
  set, not the search.** Published pruning methods report removing far more (roughly 60-90%, from memory, not
  re-checked, and they usually use importance scores and/or fine-tuning).
- **Caveats:** 3 seeds, 4 scenes (too few for a significance test), an untuned GA (mutation 0.1, pop 32+64), one fixed
  tau. On bicycle the test views are easier than the fitness views (test PSNR ~1.7 dB above fit PSNR), so a fixed
  40 dB floor is conservative there. Two grid-12 solutions land at 39.92 and 40.00 dB on the test views.
- The comparison scripts were scratch (not in the repo) and ran with `gpu-run` on the 5070 Ti. Reimplement them as
  `scripts/` tools when the experiment loop is built.

**Planned directions (not implemented; ordered by expected impact).** All three follow from the result above: the
EA only has room to matter where a sweep cannot cover the search space.

1. **Importance-score gene.** One per-scene pass computes a per-Gaussian importance (e.g. accumulated
   transmittance-weighted opacity or contribution over the train views; `render.py` already computes per-Gaussian
   contributions); the gene is a threshold on its percentile, ANDed with the existing genes (Q5).
   *Why:* it is where published methods get their gains, and it is the likeliest way to get below 65-74% kept.
   *Test:* rerun the comparison above with 4 genes. *Success:* kept % clearly below the 1D-opacity numbers at the
   same constraint AND the GA beating a 1D sweep over the importance gene alone. *Risk:* importance may also reduce
   to one dominant gene, in which case the EA again adds little. Needs the one-off importance pass per scene.
2. **Pareto front of size vs quality** (NSGA-II, or MAP-Elites over the compression pipeline) instead of one
   constrained point. *Why:* one scalar constraint is easy for a sweep; a front is the headline plot and where an
   EA is a natural fit. *Test:* hypervolume of the evolved front vs the front traced by the 1D opacity sweep (it
   traces a frontier for free, so it is the baseline to beat) and vs LightGaussian-style percentile pruning.
   Revisit Q1; the cross-method comparison then needs hypervolume rather than final kept %.
3. **Per-scene calibration of the quality floor.** State the constraint as an allowed real loss vs the photos
   (e.g. 0.5 dB) and derive the vs-unpruned threshold per scene from G0 (see "How to measure fitness"), instead of
   a fixed 40 dB. *Why:* a fixed 40 dB meant different real losses per scene (bicycle's test views ended ~1.7 dB
   above the fitness views), so kept % is not comparable across scenes or with papers. *Test:* check that real loss
   vs the photos on the held-out views lands near the target on all four scenes. *Cost:* one G0 measurement per
   scene on the train views; the rule-of-thumb formula runs 1.3-2x low, so validate it per scene.

**Baselines to report from now on:** random search at the same budget, a 1D opacity sweep (64 evals), a 3-gene
grid, percentile pruning at equal Gaussian count, and CMA-ES. If the GA cannot beat the 1D sweep in a richer
space, the claim is about the search space, not the algorithm.

## Importance gene, four scenes, mean-only fitness (2026-11-17)

`python -m easplatting.experiment --scene <s> --genes 4 --seeds 3` (tau 40 dB vs unpruned on 8 half-res train views;
GA 32 parents + 64 children x 10 generations = 672 evals; sweeps 64 evals per gene with the others off). The new gene
keeps Gaussians whose importance percentile is at least p; importance = per-Gaussian blend weight over all train views
(`render.gaussian_contributions`, cached in `results/cache/`).

% of Gaussians kept (lower is better); GA = mean over 3 seeds (best); random = mean over 3 seeds:

| Scene | GA, 4 genes | Random | **1-D importance sweep** | 1-D opacity sweep | GA, 3 genes (11-13) |
|---|---|---|---|---|---|
| bonsai | 45.4 (43.5) | 60.2 | **42.9** | 70.6 | 71.5 |
| counter | 45.7 (42.6) | 52.6 | **42.9** | 65.3 | 65.7 |
| garden | 55.2 (49.1) | 63.5 | **49.2** | 65.6 | 66.4 |
| bicycle | 52.3 (50.2) | 75.4 | **50.8** | 74.1 | 74.2 |

Held-out test PSNR vs unpruned (fitness constraint was 40 dB): GA bests 39.3-41.2 on bonsai, counter, bicycle; on garden
**39.6 / 33.8 / 34.2**. The importance-only sweep on garden: fit 40.2, **test 33.7**.

- **The gene is the big win and it is robust:** 43-51% kept vs 65-74% for the best opacity threshold, on all four
  scenes, and the best importance cutoff is nearly the same everywhere (percentile 0.45-0.57).
- **The GA never beats the importance-only sweep** (best seed per scene is 0.1-0.7 points under it; the means are 1.6-6
  over). The evolved genomes do combine genes (opacity ~0.01-0.06 + max scale + importance ~0.5, anisotropy off), but
  that buys at most ~1 point. The 1-D sweep covers the space.
- **Importance solutions overfit the fitness views** more than opacity ones: gap (fit - test PSNR) is +0.4 to +1.3 dB
  everywhere and +6.3 dB on garden, vs about zero or negative for opacity-only. Not yet diagnosed (see next steps).
- Fixed in passing: `ea.py` deep-copied each parent, which also copied the fitness function (and any logger); it now copies shallowly.
- 3 seeds, 4 scenes, fixed tau. **Superseded by the next section:** these kept % hide badly damaged views (garden's is not a valid 40 dB-on-test number).

## Robust fitness, four scenes (2026-11-17)

Why: the 8-view mean hid badly damaged views. Diagnosis (garden, importance solutions): mean PSNR over **all** train views
is 40.4 dB (the 8 views are fine on average), but the **worst train view is 24 dB** and the worst test view 21 dB, so test
mean falls to 33.7 dB. More views alone (32, mean only) did not help (test 33.9-34.5). What helped: 32 fitness views plus a
**worst-view floor of 35 dB** (`--views 32 --worst-tau 35`).

% kept at mean >= 40 dB and worst fitness view >= 35 dB (GA = mean over 3 seeds (best); sweep = 64 evals, 1 gene):

| Scene | GA | Random | Importance sweep | Opacity sweep | Test PSNR, GA (worst view) |
|---|---|---|---|---|---|
| bonsai | 61.7 (52.2) | 67.6 | **50.8** | 73.9 | 40.7-44.8 (32-36) |
| counter | **54.2 (52.4)** | 63.8 | 55.6 | 65.3 | 40.1-41.5 (34-35) |
| garden | **58.6 (57.4)** | not run | 63.5 | 67.2 | 37.2-38.2 (24-25) |
| bicycle | 66.2 (60.5) | 78.7 | **60.3** | 74.1 | 40.9-45.3 (34-37) |

- **Honest cost of robustness:** 8-14 points more kept than the 8-view numbers (importance sweep 43/43/49/51 -> 51/56/64/60),
  still 4-23 points better than opacity alone. The earlier 43-51% figures were not safe to report.
- **GA vs sweep is now mixed:** the GA wins on counter and garden (by 1.4 and 4.9 mean), loses on bonsai and bicycle. Caveat:
  the 64-point sweep overshoots the sharp feasible edge (best point sits at 46-50 dB mean), so a finer sweep may erase the
  GA wins (next step 2). GA means are noisy (bonsai 52-74% across seeds): the thin feasible region needs more evals or a
  better mutation size.
- **The floor does not fully transfer:** garden's test worst view is still ~25 dB; elsewhere 32-37 dB. Importance is
  the damaging gene (opacity-only solutions have test worst 37-39 dB on garden).
- 3 seeds per method, one tau, one floor (35 dB, chosen from opacity solutions' worst views of ~37 dB; not tuned).

## Compute budget

**Datasets by purpose.** Dev/debug of the EA loop: NeRF-Synthetic (lego) or a 7k-step gsplat truck
(whole runs in seconds; not for results). Main results: Mip-NeRF 360, 3-4 scenes, indoor first
(bonsai, counter), then garden / bicycle (comparable with LP-3DGS, Speedy-Splat, PUP-3DGS,
LightGaussian). Optional secondary table: the 8 NeRF-Synthetic scenes (cheap, but no unbounded
background, which is where scene-specific thresholds should matter most).

**Measured** (RTX 5070 Ti, `cost` rows re-run 2026-11-13, within ~3% of the 2026-11-06 run; reproduce with `python scripts/measure.py cost`, and re-run it on any new GPU type). One evaluation = mask -> prune -> render 8 train views at half res -> PSNR vs cached baselines.

| | Result |
|---|---|
| Cost per evaluation | **~9-10 ms per million Gaussians** (8 views, half res), linear in N: bonsai 1.25 M = 12.5 ms, truck 2.05 M = 19.7 ms, tiled 5 M = 46 ms, tiled 10 M = 89 ms |
| Memory | ~1.1 GB per million Gaussians at 8 views per call (10 M = 11.3 GB of the 16 GB card); lower `batch_size` beyond that |
| GPU sharing | 1 / 2 / 4 processes on one GPU = 74 / 69 / 72 evals/s aggregate: **one process saturates the GPU**; throughput scales with GPU count, not processes |
| Views / resolution | 16 views = 2x cost; half res saves ~35%, quarter res ~45% (cost is per-Gaussian projection and sorting, not pixels) |
| Metrics in the loop | PSNR +0.25 ms; SSIM +8 ms (+65%); LPIPS-alex +13 ms (2x); **LPIPS-VGG +194 ms (16x)**: keep it for the final eval |
| Cheap vs full fitness | 4-16 views at 0.25-0.5 scale rank 40 random genomes with Spearman 0.995-0.998 vs 64 views at full res |
| Outdoor scenes | not measured; estimate ~50-60 ms/eval at the ~5-6 M Gaussians the outdoor scenes have, from the linear scaling above |

**What a throwaway prototype showed** (bonsai; pop 100 = 2 elites + 98 offspring per generation, tournament
k=3, Gaussian mutation with sigma = 10% of each gene's range; fitness = kept fraction, feasibility-first, subject
to >= 40 dB vs the unpruned render on 8 half-res train views; 4,020 evals = 50 s per run, 12.4 ms/eval):
- All 3 GA seeds reach the same optimum: **70.5% of Gaussians kept at 40.0-40.1 dB** (right on the constraint).
  They are within half a point of it by generation ~10 (~1,000 evals); the rest is polishing.
- Random search with the same ~4,000 evals ends at 72.9 / 75.1 / 73.7% kept: **the GA wins by 2-5 points**, and
  random search is still moving. Modest but real; on 3 smooth genes it is not a landslide.
- Correction (2026-11-11): the first version of this prototype treated the top of the anisotropy range as "off".
  It is the 99.9th percentile, so every genome silently removed 1,219 extreme needle-like Gaussians (0.1%) and
  quality could never exceed ~40 dB. That produced a false "all runs infeasible" and a false "GA ~ random search /
  need an importance gene". **A gene's "off" must be exactly reachable** (viewer: slider ends mean inf / 0;
  prototype: outer 5% of the range decodes to "off"). The genome decoder must do the same.
- What it does say about the 3 threshold genes on bonsai: they remove ~30% of Gaussians for ~1 dB real loss vs the
  photos, and ~11% for free. Whether an importance-score gene reaches smaller sizes at equal quality is an open
  question to test, not something this data shows.

**Budget** (from the above; evals/s ~ 1000 / (10 ms per M Gaussians x N + ~2 ms)): an indoor-scene run of
10,000 evals is ~2 min, an outdoor-scene run ~8-10 min. Because the GA converges in ~1,000-3,000 evals,
budget ~3,000 evals/run (indoor ~40 s, outdoor ~3 min). A study of 2 indoor + 2 outdoor scenes x 3 stochastic
methods (GA, random, CMA-ES) x 30 seeds = 360 runs is ~10 GPU-hours at 3,000 evals, ~35 at 10,000.

**Making evaluations cheap (by expected impact)**
1. Cheap fitness, full final eval: fixed 8-16 TRAIN views at reduced resolution while evolving;
   only final/best genomes at protocol resolution on TEST views.
2. Validate the shortcut: ~50 random genomes, cheap vs full fitness; require Spearman rho >= 0.95.
3. PSNR/MSE only in the loop (`metrics.psnr_fast` vs cached baselines); SSIM/LPIPS at final eval.
4. Load + activate the scene once per process, cache baselines, run several seeds per process.
5. All fitness views of an individual in one `render_views` call; tune `batch_size`.
6. Fitness cache keyed on the (rounded) genome so elites/duplicates aren't re-rendered.
7. Later: render several individuals per call by zeroing opacities (gsplat 1.5.3 culls opacity
   < 1/255 and accepts `opacities` of shape `[B, N]`). Memory-hungry, and since one process already
   saturates the GPU it only saves launch overhead; FPS must still be measured on an actually pruned scene.
   The prune copy (~1.4 ms per M Gaussians, ~13% of an eval) can be skipped by zeroing opacities instead.
8. Optional multi-fidelity: score all on 4 views, re-score the top quarter on the full set.
9. Don't rotate fitness views per generation (noisy, elites not comparable).

## How to measure fitness (assessment, 2026-11-11)

Evidence: `python scripts/measure.py metrics` (bonsai, 8 train views, full res; writes pictures to
`results/metrics_intuition/`: `levels.png` = what each quality level looks like, `frontier.png` = the curves).

**What the metrics mean here** (all compare a pruned render with the unpruned one, which is the reference):
- **PSNR** = 10 log10(1/MSE): the energy of the pixel errors, on a log scale. 60 dB = RMS error 0.1% of the
  range (0.3/255), 50 dB = 0.3%, 40 dB = 1% (2.5/255), 30 dB = 3.2% (8/255), 20 dB = 10%. Every +10 dB is 10x less
  error energy. Blind to *where* or *what kind*; big errors on bright edges dominate. Infinite for identical
  images, so cap it (60 dB).
- **SSIM** (0-1): compares local brightness, contrast and structure in small windows. Sits near 1 for small
  damage (0.999, 0.997, 0.979 ...), so the signal is in the third decimal. Reacts to lost texture.
- **LPIPS** (0 = identical): distance between deep-network features, a proxy for "does it look different to a
  trained eye". Catches texture/blur differences PSNR forgives. Costs 12 ms (alex) or 184 ms (vgg) per eval.
- **Gaussians kept** (%): the size side. Bytes are proportional to it while no SH is dropped.

**What pruning does to quality** (opacity threshold alone, bonsai): the first ~11% of Gaussians are free
(opacity below 1/255 is never rendered, so PSNR vs unpruned is exactly infinite). After that quality falls
quickly: 76% kept = 49 dB (visually identical), 72% = 41 dB, 60% = 35 dB (texture visibly thinner), 45% = 31 dB,
32% = 26 dB (streaks, dark patches).

**What "vs unpruned" means for the real result** (loss vs the photos; the model itself is 31.7 dB vs photos):

| PSNR vs unpruned | mean kept | real loss vs photos (dB PSNR) | SSIM loss | LPIPS-alex rise |
|---|---|---|---|---|
| 60+ dB | 94% | 0.002 | 0.0000 | 0.0000 |
| 50-60 | 86% | 0.09 | 0.0003 | 0.0001 |
| 45-50 | 92% | 0.20 | 0.0007 | 0.0006 |
| 40-45 | 84% | 0.76 | 0.0016 | 0.0013 |
| 35-40 | 83% | 1.5 | 0.005 | 0.004 |
| 30-35 | 66% | 3.7 | 0.033 | 0.023 |

Rule of thumb: pruning error adds *in power* to the model's own error, so real loss L = 10 log10(1 + 10^((G0 - U)/10)),
with G0 = model PSNR vs photos and U = PSNR vs unpruned. It has the right shape but runs 1.3-2x low in the data
(errors overlap: both live in textured regions), so for an allowed real loss of 0.5 dB use U >= ~42-43 dB on bonsai
(formula alone says 40.8). The two PSNRs differ per scene, so **state the constraint as a real loss and derive the
vs-unpruned threshold per scene** from G0 (measured once per scene on the train views).

**All metrics rank pruned genomes almost identically** (Spearman over 88 genomes: PSNR vs LPIPS-vgg 0.996, vs
LPIPS-alex 0.993, vs SSIM 0.975), and PSNR vs unpruned predicts the real loss best (rank correlation 0.996,
LPIPS-vgg 0.991, LPIPS-alex 0.987, SSIM 0.978). The cheap loop fitness (8 views, half res) ranks like the
full-resolution one at 0.998. Pruning damage is one kind (missing Gaussians), so extra metrics add cost, not signal.

**Recommendation**
1. **In the loop:** PSNR vs the cached unpruned renders (cap 60 dB) on 8 fixed train views at half res, plus Gaussian
   count. 12 ms per evaluation; nothing else in the loop.
2. **Objective:** feasibility-first: minimize kept fraction subject to U >= U_min, where U_min comes from an allowed
   real loss (e.g. 0.5 dB, `U_min = G0 - 10 log10(10^(L/10) - 1)` plus ~2 dB of margin). Sweep L for a quality-size
   curve; NSGA-II would give that curve in one run (decision Q1 below).
3. **Final report per solution** on the held-out test views at full res: PSNR/SSIM/LPIPS-vgg vs the photos *and*
   vs unpruned, kept %, bytes, FPS. The photos column is what papers report.
4. **Guards:** floor the free region (the baseline "opacity 0.005 cull" already keeps 88% at ~92 dB); log the
   incumbent's test-view PSNR every ~10 generations (never used for selection); the mean over views can hide one
   badly damaged view, so consider also constraining the worst view (not measured yet).
5. **Caveat:** one scene, one 8-view set, one gene family; re-run the script per scene before trusting the
   rule-of-thumb numbers elsewhere.

## Decisions to make next (parked 2026-11-06)

Four questions block implementation of `genome.py` / `fitness.py`; we haven't answered them yet. Leans
are mine. Full list of everything else is under "Open questions" below.

1. **Fitness form (Q1).** (How to *measure* quality is settled above; this is about how to combine it with size.) (a) *Constrained, feasibility-first (lean):* minimize kept Gaussians subject to
   PSNR-vs-unpruned >= tau; feasible always beats infeasible; one scalar, so GA vs random vs CMA-ES stats
   stay simple; sweep tau for a quality-size curve. (b) *NSGA-II:* whole Pareto front in one run (pop 100
   suits it), but cross-method comparison needs hypervolume. (c) *Weighted sum + lambda sweep:* simplest,
   but the weight silently decides the trade-off (it did in the prototype).
2. **Genes in the first version (Q4).** *Update 2026-11-13: the 3-threshold version is now tested and reduces to
   an opacity threshold on all four scenes, so move the importance-score gene (b) up; see "Findings so far".*
   (a) *3 thresholds first, importance score next (earlier lean):* opacity,
   max scale, anisotropy already exist (`thresholds.keep_mask`). (b) Importance score from the start: likely
   better compression, needs a per-scene importance pass first. (c) 3 genes + SH-degree cap: then size
   must be measured in bytes.
3. **Scenes (Q12).** (a) *Indoor first, outdoor after (lean):* bonsai + counter (~2 min per 10k-eval run),
   then garden + bicycle (~8-10 min). (b) All four from the start (~3x compute up front). (c) Another set.
4. **Fine-tuning after pruning (Q11).** (a) *Training-free loop, fine-tune chosen finals only (lean):*
   ~1-1.5 min per solution, reported separately, for comparison with LP-3DGS / Speedy-Splat. (b) Strictly
   training-free. (c) Fine-tune every final solution (~2x study cost).

**Already settled by the 2026-11-06 measurements (object if you disagree):** 8 fitness views at half
resolution (Q8; Spearman 0.998 vs full); budget by evaluation count, also report wall time (Q10); pop 100
but ~20-30 generations with a convergence check (Q15; the GA converged by gen ~10); feasibility-first
instead of a fixed penalty (Q3); size = Gaussian count, also report bytes (Q2); a Gaussian is kept only
if all conditions pass (Q5); accept the 0.008 LPIPS gap (Q16).

## Open questions

**Objective**
1. **Fitness form?** Weighted sum (needs a λ sweep), constrained (minimize size s.t. ΔPSNR ≤ τ), or
   NSGA-II. *Lean:* constrained for the main GA-vs-baselines comparison (one scalar, easy stats);
   NSGA-II as an extension since its Pareto front is the headline plot anyway.
2. **What is "size"?** Gaussian count, or bytes? They differ once an SH-degree-cap gene exists.
3. **Infeasible genomes** (empty or near-empty mask, ΔPSNR way over τ): penalty, repair, or reject?

**Genome**
4. **Which genes?** Notes §6.1 lists six. Start with opacity + max-scale (+ anisotropy)? The
   importance-score gene needs a one-off per-scene pass; the SH cap changes the representation.
5. **How do genes combine?** Keep = all conditions pass (AND) is the obvious reading; confirm.
6. **Bounds from percentiles per scene, or fixed global bounds?** Per-scene bounds make "scene-specific
   thresholds" partly an artifact of the bounds.
7. **Max-scale hazard:** unbounded scenes (garden, bicycle) use huge Gaussians for background. Let the
   EA discover this, or clamp the gene's range?

**Evaluation**
8. **Fitness views:** how many (8? 16?), what resolution, fixed per run? Validate the shortcut with
   ρ ≳ 0.95 vs full fitness (item 2 above) before trusting it.
9. **Overfitting check:** log held-out test PSNR of the incumbent every K generations (never used
   for selection)? Costs little, tells us if evolved thresholds overfit the fitness views.
10. **Budget fairness:** equal number of evaluations, or equal wall time? Pruned individuals render
    faster, so these differ.

**Scope / experiments**
11. **Fine-tune after pruning?** Strictly training-free is cheaper and cleaner; fine-tuning matches
    LP-3DGS / Speedy-Splat. *Lean:* prune-only in the loop, optional fine-tune of the final solution.
12. **Scenes:** which 3-4 Mip-NeRF 360 scenes (bonsai + counter indoor, garden/bicycle outdoor)? *Update
    2026-11-13:* those four are downloaded and validated locally. For a paired EA-vs-baseline test, consider adding
    kitchen, stump (and our own T&T/Deep Blending splats trained on the train split).
13. **Per-scene vs transferable:** does a genome evolved on one scene work on another? Cheap to test
    and a stronger story than per-scene numbers alone.
14. **Baselines:** random search (same budget), LightGaussian-style percentile pruning, CMA-ES
    (optional). With 3-6 smooth genes, random search may match the GA. Are we fine reporting that?
    *Update 2026-11-13:* random search did not match the GA (3-6 points worse), but a 64-eval 1D opacity sweep did.
    Add the 1D sweep and a grid to the baselines (see "Findings so far").
15. **Seeds / pop / gens:** 20-30 seeds; pop 32 x 50 gens should be plenty. Any reason for more?

**Housekeeping**
16. Our LPIPS-VGG is 0.008 above NerfBaselines' (within tolerance). Worth chasing, or accept?
17. Who owns what (genome/fitness vs. logging/plots vs. run scripts)?

## Feedback and visualization ideas

Everything below hangs off one rule from the notes: **one JSONL row per evaluation** (run id, seed,
generation, genome, kept count, ΔPSNR, wall time, cache hit). Plots are then post-processing.

**During a run**
- **Terminal progress line** per generation: gen, best fitness, kept %, ΔPSNR, evals/s, ETA, GPU mem.
- **TensorBoard** (already installed with gsplat's trainer deps): scalars live, plus a render strip
  (GT | unpruned | best pruned | error) every K generations.
- **`progress.png`** rewritten every N generations: best/median fitness vs evaluations, kept
  fraction, and the population in the (size, ΔPSNR) plane coloured by generation.
- **Gene trajectories**: per-gene population histogram or violin over generations; shows convergence
  and which genes the EA actually uses.
- **Health warnings**: empty-mask rate, fitness-cache hit rate, train-vs-test gap of the incumbent.

**At the end of a run / study**
- **Convergence curves**: best fitness vs evaluations, median ± IQR over seeds, GA vs random vs CMA-ES.
- **Pareto plot**: kept fraction vs test PSNR, baselines (unpruned, opacity 0.005, LightGaussian-style)
  as points, evolved solutions as a cloud/front.
- **Fitness landscape slice**: grid-evaluate two genes (e.g. opacity x max-scale) on bonsai. Cheap,
  and shows whether the landscape is smooth/monotone, which predicts whether the GA beats random.
- **Thresholds on distributions**: opacity / scale histograms with evolved thresholds overlaid.
- **"What got removed"**: render only the pruned-away Gaussians next to the kept scene, and a top-down
  point plot coloured kept/removed. Most intuitive picture for a talk.
- **Where it hurts**: per-test-view ΔPSNR bars, and camera poses coloured by ΔPSNR.
- **Threshold sweep GIF**: orbit or fixed view while opacity threshold rises 0 -> 0.2. Good for slides.
- **Single-page run report** (`summary.html`: config, curves, tables, image strips) generated per
  run/study; can also be published as a shareable page.

**Interactive**
- **`viewer.py`** (done): threshold sliders, kept count / size / FPS / view PSNR, and views of
  pruned, unpruned, removed-only and error. Next: load an evolved genome JSON into it.

*Lean for a first pass:* JSONL logging + terminal line + `progress.png` + TensorBoard, then Pareto
and convergence plots from the JSONL; landscape slice and removed-Gaussians render for the write-up.
