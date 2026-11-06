# EA plan: compute budget, open questions, feedback ideas

For Eli + Areeb as we write the EA. "Lean" = a suggestion, not a decision. Background and fitness
design: `docs/GS_PRUNING_NOTES.md` §5-8. Already decided (see `HANDOFF.md`): unpruned renders as
the in-loop reference, evolve on train views / report on test, PSNR-only in the loop,
NerfBaselines protocol.

## Compute budget

**Datasets by purpose.** Dev/debug of the EA loop: NeRF-Synthetic (lego) or a 7k-step gsplat truck
(whole runs in seconds; not for results). Main results: Mip-NeRF 360, 3-4 scenes, indoor first
(bonsai, counter), then garden / bicycle (comparable with LP-3DGS, Speedy-Splat, PUP-3DGS,
LightGaussian). Optional secondary table: the 8 NeRF-Synthetic scenes (cheap, but no unbounded
background, which is where scene-specific thresholds should matter most).

**Cluster:** see `docs/CLUSTER.md` (VACC GPUs, env, preemption, one GPU SKU for timings).

**Measured** (RTX 5070 Ti, 2026-11-06; reproduce with `python scripts/bench_eval.py`, and re-run it on each
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

**What a throwaway prototype showed** (bonsai; pop 100, 100 offspring per generation, tournament k=3, Gaussian
mutation, 2 elites, 100 generations = 9,900 evals = 123 s per run; fitness = kept fraction + 0.05 per dB
below a 40 dB-vs-unpruned constraint):
- The GA **converged by generation ~10 (~1,000 evals)** in all 3 seeds, to nearly the same optimum
  (fitness 0.7768-0.7769); the remaining 90 generations changed nothing.
- Random search with the same 9,000 evals ended 1% worse (0.783-0.785) and was still creeping; with 1,000
  evals it was 0.5-3% worse. The 3-gene landscape is easy, so **GA ~ random search is a likely finding.**
- All three GA runs ended *infeasible* (39.3 dB < 40 dB): a fixed penalty weight lets the optimizer trade the
  constraint away. Use feasibility-first ranking (infeasible always loses to feasible) or a lambda sweep.
- With only opacity / max-scale / anisotropy thresholds, ~39 dB vs unpruned kept ~74% of Gaussians. Treat
  that as a hint that an importance-score gene (LightGaussian-style) may be needed for headline compression.

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

## Decisions to make next (parked 2026-11-06)

Four questions block implementation of `genome.py` / `fitness.py`; we haven't answered them yet. Leans
are mine. Full list of everything else is under "Open questions" below.

1. **Fitness form (Q1).** (a) *Constrained, feasibility-first (lean):* minimize kept Gaussians subject to
   PSNR-vs-unpruned >= tau; feasible always beats infeasible; one scalar, so GA vs random vs CMA-ES stats
   stay simple; sweep tau for a quality-size curve. (b) *NSGA-II:* whole Pareto front in one run (pop 100
   suits it), but cross-method comparison needs hypervolume. (c) *Weighted sum + lambda sweep:* simplest,
   but the weight silently decides the trade-off (it did in the prototype).
2. **Genes in the first version (Q4).** (a) *3 thresholds first, importance score next (lean):* opacity,
   max scale, anisotropy already exist (`viewer.keep_mask`). (b) Importance score from the start: likely
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
