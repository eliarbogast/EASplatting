# EA plan

Status, decisions, next steps, the compute budget, how to measure fitness, and the open design questions, for
Eli + Areeb as we write the EA. "Lean" = a suggestion, not a decision. How to run things: [`RUNNING.md`](RUNNING.md).
Background and fitness-design options: [`BACKGROUND.md`](BACKGROUND.md) sections 5-8.

## Status and next steps (2026-11-11)

**Done and verified on a real GPU** (RTX 5070 Ti; results in [`RUNNING.md`](RUNNING.md) section 7): scene loading,
cameras, rendering, metrics (match NerfBaselines and gsplat's own eval), the evaluation report, the interactive
viewer, the threshold -> mask logic (`easplatting/thresholds.py`), data/setup/measurement scripts. 37 CPU tests.
**First EA test works** (branch `ea-test`, built on Areeb's draft): genome decoding, feasibility-first fitness and a minimal
mutation + tournament loop; `python -m easplatting.run_ea` reaches 71% of bonsai's Gaussians kept at 40.3 dB in 5 generations (4 s).
**Not started:** random-search baseline, config-driven runs and logging, the full experiment loop, the VACC job scripts.

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

**Next steps**
1. Settle the four decisions under "Decisions to make next" below (fitness form, genes, scenes, fine-tuning).
2. `easplatting/genome.py`: gene bounds in log space with an explicit "off" zone at the range ends; decode to a mask
   with `thresholds.keep_mask`.
3. `easplatting/fitness.py`: mask -> render the 8 fitness views -> PSNR vs cached baselines + size; feasibility-first;
   short-circuit empty masks; cache by rounded genome. See "How to measure fitness".
4. Random search with the same evaluation budget first (validates the loop), then the GA (tournament + Gaussian
   mutation + elitism; BLX/SBX crossover optional).
5. VACC: one process per (scene x method), seeds sequentially, SLURM array over those, persistent
   `TORCH_EXTENSIONS_DIR` ([`RUNNING.md`](RUNNING.md) section 8).

## Compute budget

**Datasets by purpose.** Dev/debug of the EA loop: NeRF-Synthetic (lego) or a 7k-step gsplat truck
(whole runs in seconds; not for results). Main results: Mip-NeRF 360, 3-4 scenes, indoor first
(bonsai, counter), then garden / bicycle (comparable with LP-3DGS, Speedy-Splat, PUP-3DGS,
LightGaussian). Optional secondary table: the 8 NeRF-Synthetic scenes (cheap, but no unbounded
background, which is where scene-specific thresholds should matter most).

**Cluster:** see [`RUNNING.md`](RUNNING.md) section 8 (VACC GPUs, env, preemption, one GPU SKU for timings).

**Measured** (RTX 5070 Ti, 2026-11-06; reproduce with `python scripts/measure.py cost`, and re-run it on each
VACC GPU type). One evaluation = mask -> prune -> render 8 train views at half res -> PSNR vs cached baselines.

| | Result |
|---|---|
| Cost per evaluation | **~9-10 ms per million Gaussians** (8 views, half res), linear in N: bonsai 1.25 M = 12.7 ms, truck 2.05 M = 19.8 ms, tiled 5 M = 45 ms, tiled 10 M = 87 ms |
| Memory | ~1.1 GB per million Gaussians at 8 views per call (10 M = 11.4 GB); lower `batch_size` beyond that |
| GPU sharing | 1 / 2 / 4 processes on one GPU = 76 / 71 / 74 evals/s aggregate: **one process saturates the GPU**; throughput scales with GPU count, not processes |
| Views / resolution | 16 views = 2x cost; half res saves ~35%, quarter res ~45% (cost is per-Gaussian projection and sorting, not pixels) |
| Metrics in the loop | PSNR +0.25 ms; SSIM +8 ms (+60%); LPIPS-alex +12 ms (2x); **LPIPS-VGG +184 ms (16x)**: keep it for the final eval |
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
2. **Genes in the first version (Q4).** (a) *3 thresholds first, importance score next (lean):* opacity,
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
12. **Scenes:** which 3-4 Mip-NeRF 360 scenes (bonsai + counter indoor, garden/bicycle outdoor)?
13. **Per-scene vs transferable:** does a genome evolved on one scene work on another? Cheap to test
    and a stronger story than per-scene numbers alone.
14. **Baselines:** random search (same budget), LightGaussian-style percentile pruning, CMA-ES
    (optional). With 3-6 smooth genes, random search may match the GA. Are we fine reporting that?
15. **Seeds / pop / gens:** 20-30 seeds; pop 32 x 50 gens should be plenty. Any reason for more?

**Housekeeping**
16. Our LPIPS-VGG is 0.008 above NerfBaselines' (within tolerance). Worth chasing, or accept?
17. Who owns what (genome/fitness vs. logging/plots vs. VACC scripts)?

## Feedback and visualization ideas

Everything below hangs off one rule from the notes: **one JSONL row per evaluation** (run id, seed,
generation, genome, kept count, ΔPSNR, wall time, cache hit). Plots are then post-processing.

**During a run**
- **Terminal progress line** per generation: gen, best fitness, kept %, ΔPSNR, evals/s, ETA, GPU mem.
- **TensorBoard** (already installed with gsplat's trainer deps): scalars live, plus a render strip
  (GT | unpruned | best pruned | error) every K generations. Works over a port-forward on the VACC.
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
