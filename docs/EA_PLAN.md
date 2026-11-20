# EA plan

Status, decisions, next steps, the compute budget, how to measure fitness, and the questions still open, for
Eli + Areeb. How to run things: [`RUNNING.md`](RUNNING.md). Results: [`figures/README.md`](../figures/README.md).
Background and fitness-design options: [`BACKGROUND.md`](BACKGROUND.md) sections 5-8. The early open questions, the
parked decisions and the first prototype are in [`archive/ea_plan_2026-11.md`](archive/ea_plan_2026-11.md).

## Status (2026-11-20)

**Built and verified.** On a real GPU (RTX 5070 Ti): scene loading, cameras, rendering, metrics (match NerfBaselines and gsplat's own
eval), the evaluation report, the interactive viewer, the threshold -> mask logic (`thresholds.py`), data/setup/measurement scripts.
**EA pipeline on `main`:**
- genome decoding (3 thresholds + an importance-percentile gene, each with an explicit "off" zone) and feasibility-first fitness (mean-PSNR
  floor, optional worst-view floor, `--views K`): `genome.py`, `fitness.py`;
- searchers in one style and interface: mutation + tournament GA (`ea.py`), CMA-ES, NSGA-II, MAP-Elites, random search, 1-D sweeps with bisection
  refinement; importance scores `sum`, `max`, `density`, `lg` (`experiment.importance_scores`, from `render.py`'s per-Gaussian contributions) and `learned`, `fisher` (`scores.py`);
- tools: `experiment.py` (constrained % kept, one JSONL row per evaluation, held-out PSNRs per best genome), `front.py` (size-quality fronts,
  hypervolume), `plot_run.py` / `plot_front.py` (figures), `strip.py` (renders), all published into `figures/`.
CPU tests run in GitHub Actions on every push (`.github/workflows/tests.yml`; `requirements.txt` must list every third-party import).
**Results so far** (details and caveats in `figures/README.md`):
- The importance gene is the big win: 43-51% kept (mean PSNR >= 40 dB) vs 65-74% for the best opacity threshold; once the worst view is
  constrained (>= 35 dB) the honest number is 51-66%.
- The EA does not earn its place yet: the GA beats random search by 9-21 points but loses to CMA-ES and the refined 1-D importance sweep on every
  scene (one basin, reached by a single importance cutoff). NSGA-II and MAP-Elites lose to the importance sweep's own front.
- **Half-res fitness over-reports quality (2026-11-20).** Re-scoring the main runs' genomes at full res shows the 40 / 35 dB floors fail on counter, garden and bicycle
  (garden worst view 29 dB, bicycle 27 dB). Evolving on full-res fitness views (10 seeds) costs +1 point kept indoors and +11 to +13 outdoors; importance beats opacity by 20-23 points on bonsai and
  counter but only 6 on bicycle and ~2 (a tie) on garden. GA < random everywhere and CMA-ES < GA, CMA-ES ties the refined sweep. **Full-res fitness is now the main setting** (`experiment.py --fit-scale 1`); on garden / bicycle full-res importance scores (`--score-scale 1`) are needed too (CMA-ES 57.8 / 55.7 % kept).
- **Harder-search branch** (`explore/harder-search-2026-11-19`, not merged; bonsai + counter only, half-res fitness, 2 seeds, no significance tests; write-up in that branch's `figures/README.md`):
  - A genome of weights over 10 per-Gaussian feature ranks (max, opacity, volume, views, camdist, sum, density, lg, fisher, learned) plus per-region cutoffs contains the 1-D sweep and goes past it:
    bonsai 51.0 -> ~40 % kept, counter 42.7 -> ~37. Random search does not find it, GA and CMA-ES do (CMA-ES better by 1-3 points, no warm start needed). This is the first setting where the EA earns its place.
  - Evolved genomes miss the floors on held-out views (bonsai: new poses; counter: half-res fitness vs full-res test, fixed by a +1 dB margin).
  - Only the 10-feature genome transfers across scenes; depth-shell genomes do not (shells only help bonsai). At matched size on test views it beats max / sum / lg on mean PSNR but only ties max on the worst view.
  - Features added later (tail, edge, footprint, depth, knn, scale / anisotropy, leave-out) were within noise: the set is saturated. Fixing the gate learner (loss rescale) and a Jacobian Fisher score still left them weak (good mean, bad worst view).
  - FPS: 1.3-1.6x at ~40 % kept; Gaussian count is a poor speed proxy (opacity pruning is faster at equal count).
- The `learned` and `fisher` scores failed in this simple form (worse than opacity alone), so there is no honest comparison with published scores yet.

## Decisions already made
- **Renderer: gsplat 1.5.3** (latest release; 1.6.0 on main is unreleased with a different trainer API). Batched
  multi-camera forward renders from Python; a mask is just tensor indexing.
- **Benchmark: NerfBaselines Mip-NeRF 360 protocol**: official `images_4` (outdoor) / `images_2` (indoor), llffhold-8
  test split (every 8th name-sorted camera), LPIPS-VGG on [-1,1]-rescaled inputs. Unpruned baseline = NerfBaselines'
  `gaussian-splatting` checkpoints (trained on the train split only, published per-scene metrics). Inria's own
  pretrained models may include test views, so use them only for quick tests.
- **Scenes:** bonsai and counter (indoor, ~1.2 M Gaussians), garden and bicycle (outdoor, several million). bonsai is the dev scene.
  Tanks & Temples comes from Inria's `tandt_db` (not NerfBaselines' T&T, which is preprocessed differently).
- **Fitness:** feasibility-first. Minimize the kept fraction subject to mean PSNR >= tau (default 40 dB) vs the unpruned renders on K fixed
  train views (8, or 32 for the robust setting) at FULL resolution for reported results (`--fit-scale 1`; half-res fitness, the earlier default `--fit-scale 0.5`, misses the floors at full res), optionally also worst-view PSNR >= `--worst-tau`. Reference = the unpruned renders
  (cached), not the photos; final numbers report both. Evolve on train views, report on test views.
- **Genes:** opacity, max scale, max anisotropy, importance percentile; a gene is "off" in an explicit zone of its range, and a Gaussian is kept only
  if every active gene passes. A gene's "off" must be exactly reachable (an earlier prototype that treated the top of the anisotropy range as "off"
  silently removed 0.1% of Gaussians and capped quality at ~40 dB; see the archive).
- **Size = Gaussian count** (bytes only matter once an SH-degree gene exists). **Budget = number of evaluations**; wall time is logged too.
- **Importance:** per-Gaussian blend weight over the train views (`render.gaussian_contributions`), cached in `results/cache/`; `max` over views is
  the default for the main runs.
- **Baselines reported:** random search at equal budget, 1-D sweeps per gene (refined), CMA-ES, percentile pruning by the `lg` score.
- **Accepted:** our LPIPS-VGG is 0.008 above NerfBaselines' (within tolerance).

## Next steps (in order)
0. **Done 2026-11-20: importance scores at full res** (`--score-scale 1`) recover garden 63.1 -> 57.8 and bicycle 65.0 -> 55.7 % kept (CMA-ES) and restore the importance-over-opacity gap (-7 / -16). Open: bonsai / counter at full-res scores (expected small),
   mixed fitness (full-res worst-view check only), 0.75 resolution for cost.
1. **Re-run the harder search on the main protocol, then merge the branch.** The weighted-feature genome is the real EA result (see the branch bullet above) but was found at half-res fitness on two scenes with 2 seeds. Redo it with
   full-res fitness and scores (`--fit-scale 1 --score-scale 1`), a margin on the floors (or full-res worst-view check), all four scenes and 10 seeds with Mann-Whitney vs CMA-ES on 4 genes and the refined sweep. Drop depth shells and the
   extra features (no gain), keep the 10-feature set, then merge `explore/harder-search-2026-11-19` into `main`.
2. **`learned` / `fisher` scores:** still weak after the loss-rescale fix. Low priority; they stay in the feature set only because the genome can down-weight them. LightGaussian-style `lg` is our comparison baseline (no fine-tune).
3. **FPS-aware objective.** Measured on the branch: 1.3-1.6x at ~40 % kept, and count is a poor proxy (opacity pruning is faster at equal count). Add FPS or a render-cost term to the size objective and report it in the main tables.
4. **Fine-tune finals only** and a byte-size objective (SH-degree gene) for comparison with LightGaussian-style numbers.
5. **Generalization claim:** the 10-feature genome transfers across scenes (branch, bonsai <-> counter); test leave-one-scene-out on all four, with an oracle-per-scene gap.
6. **Final study design:** 20-30 seeds on the winning comparison (3 now), with the Mann-Whitney / Wilcoxon tests from `BACKGROUND.md` section 8.

**More ideas (not started)**
- *Per-view-aware importance:* score each Gaussian by its worst-view contribution (max over views, not the sum), which targets the tail damage the mean hides.
- *Importance on fitness views only*, or leave-views-out importance, to cut the train/test gap at its source.
- *Multi-fidelity:* score all genomes on 4 views, re-score the top quarter on 32; keeps the robust fitness cheap.
- *Fitness cache* keyed on the (rounded) genome, so elites and duplicates are not re-rendered (not implemented).
- *Transfer:* does a genome evolved on one scene work on another (the ~0.5 importance percentile is nearly the same everywhere).
- *More scenes* for a paired EA-vs-baseline test: kitchen, stump, or our own T&T / Deep Blending splats trained on the train split.

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
| Outdoor scenes | measured in the experiment runs (`seconds` in `summary.json`; a 672-eval GA run, 8 views): bonsai 5.9 s, bicycle 18.8 s, garden 28.2 s (~42 ms/eval); garden at 32 views 89 s (~130 ms/eval). Consistent with the linear scaling above |

**What the first prototype showed** (bonsai, 3 genes; details in [`archive/ea_plan_2026-11.md`](archive/ea_plan_2026-11.md)): GA 70.5% kept at 40 dB,
random search 73-75% at the same ~4,000 evaluations, ~12 ms per evaluation. The experiments since (`figures/README.md`) replace these numbers.

**Budget** (from the above; evals/s ~ 1000 / (10 ms per M Gaussians x N + ~2 ms)): an indoor-scene run of 10,000 evals is ~2 min, an outdoor-scene
run ~8-10 min. Our runs use 672 evals each (pop 32 + 64 children x 10 generations); the GA converges in ~1,000-3,000 evals. A study of 2 indoor + 2 outdoor
scenes x 3 stochastic methods (GA, random, CMA-ES) x 30 seeds = 360 runs is ~10 GPU-hours at 3,000 evals, ~35 at 10,000. With 32 views the cost per
evaluation is ~4x the 8-view numbers above.

**Making evaluations cheap (by expected impact)**
1. Cheap fitness, full final eval: fixed 8-16 TRAIN views at reduced resolution while evolving;
   only final/best genomes at protocol resolution on TEST views.
2. Validate the shortcut: ~50 random genomes, cheap vs full fitness; require Spearman rho >= 0.95.
3. PSNR/MSE only in the loop (`metrics.psnr_fast` vs cached baselines); SSIM/LPIPS at final eval.
4. Load + activate the scene once per process, cache baselines, run several seeds per process.
5. All fitness views of an individual in one `render_views` call; tune `batch_size`.
6. Fitness cache keyed on the (rounded) genome so elites/duplicates aren't re-rendered (not implemented).
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
   curve (`--allowed-loss` derives tau this way); `front.py` (NSGA-II / MAP-Elites) gives the curve in one run.
3. **Final report per solution** on the held-out test views at full res: PSNR/SSIM/LPIPS-vgg vs the photos *and*
   vs unpruned, kept %, bytes, FPS. The photos column is what papers report.
4. **Guards:** floor the free region (the baseline "opacity 0.005 cull" already keeps 88% at ~92 dB); log the
   incumbent's test-view PSNR every ~10 generations (never used for selection); the mean over views can hide one
   badly damaged view, so also constrain the worst view (`--worst-tau`; measured, see `figures/README.md`).
5. **Caveat:** this assessment is bonsai only (one 8-view set, one gene family); re-run the script per scene before trusting the
   rule-of-thumb numbers elsewhere.

**Fitness views in code** (`experiment.py`; `--views K` sets K, default 8, 32 for the robust setting):
```python
views = cams.subset(select_indices(len(cams), "train", k=K)).scaled(0.5)   # K evenly spaced train views, half resolution
baselines = render_views(scene, views)                                    # unpruned render of the same views, once
psnr = psnr_fast(render_views(prune(scene, mask), views), baselines, max_db=60)   # per view; mean (and min) feed the constraint
```
`Cameras.scaled(s)` scales focal lengths, principal point and image size, so gsplat renders natively at the smaller size. Both sides use
the same resolution, so resolution effects cancel. Validated on bonsai (80 genomes, half-res vs full-res): Spearman 0.997, same 40 dB
feasible/infeasible call for 98%, ~12 ms per evaluation. Guard rails: check the ranking on each new scene (Spearman >= 0.95 over ~50
random genomes); report final numbers on held-out test views at full resolution (the experiment tool does); a mean over views hides one
badly damaged view (see `figures/README.md`), so also constrain the worst; a min-scale gene would need a higher resolution.

## Open questions
Most early questions were answered by the experiments; the list and answers are in [`archive/ea_plan_2026-11.md`](archive/ea_plan_2026-11.md).
Still open:
1. **Size in bytes?** Needs an SH-degree-cap gene and the byte-size objective (next step 4).
2. **Fine-tune after pruning?** Strictly training-free is cheaper and cleaner; fine-tuning matches LP-3DGS / Speedy-Splat. *Lean:* prune-only in the loop,
   optional fine-tune of the final solution, reported separately (~1-1.5 min per solution).
3. **Transfer:** does a genome evolved on one scene work on another? Cheap to test, a stronger story than per-scene numbers.
4. **Which comparison gets the 20-30 seeds** (next step 6), and is it fine to report that CMA-ES / a refined sweep match the GA on the current 4-gene space?
5. **Fitness-floor values:** is 40 dB mean / 35 dB worst the right pair, or should every scene use `--allowed-loss`? (A fixed 40 dB means different real
   losses per scene, so kept % is not comparable across scenes or with papers.)
6. **Who owns what** (genome/fitness vs logging/plots vs run scripts).

## Feedback and visualization
Everything hangs off one rule: **one JSONL row per evaluation** (`results/<run>/log.jsonl`); plots are post-processing.

**Done**
- JSONL per evaluation, `summary.json` per run (best genome per run, held-out PSNRs), a terminal line per run and a mean-kept summary per method.
- `plots.png` per run: convergence (A), the (% kept, PSNR) plane (B), best genomes (C), the same genomes re-measured on train / test / worst views (D);
  `compare.png` across runs; `front.png` for the size-quality fronts.
- `renders.jpg` + lossless `images/` per run: unpruned vs best genomes, error maps, zoom of the worst spot.
- `viewer.py`: threshold sliders, kept count / size / FPS / view PSNR, pruned / unpruned / removed-only / error views. Next: load an evolved genome JSON into it.

**Not done (ideas)**
- TensorBoard scalars and `progress.png` rewritten every N generations; per-gene population histograms over generations; health warnings (empty-mask rate,
  train-vs-test gap of the incumbent).
- Fitness-landscape slice (grid over two genes); opacity / importance histograms with the evolved cutoffs overlaid.
- Talk visuals (side-by-side video, what-got-removed, evolution animation, ...): the ranked list is in `figures/README.md` "Presentation kit".
- Per-test-view PSNR bars and camera poses coloured by PSNR (the tail damage); a threshold-sweep GIF (opacity 0 -> 0.2).
- A single-page `summary.html` per run or study.
