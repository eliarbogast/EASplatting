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

**Estimate** (measured: bonsai renders in ~13 ms/view at 1559x1039, from `report.json`).
Fitness = ~8 train views at reduced resolution = 20-60 ms = 15-50 evals/s, faster as individuals
prune more. A run of pop 32 x 50 gens = 1,600 evals = 0.5-2 min. A study of 4 scenes x 4 methods
x 30 seeds = 480 runs = 4-16 GPU-hours.

**Making evaluations cheap (by expected impact)**
1. Cheap fitness, full final eval: fixed 8-16 TRAIN views at reduced resolution while evolving;
   only final/best genomes at protocol resolution on TEST views.
2. Validate the shortcut: ~50 random genomes, cheap vs full fitness; require Spearman rho >= 0.95.
3. PSNR/MSE only in the loop (`metrics.psnr_fast` vs cached baselines); SSIM/LPIPS at final eval.
4. Load + activate the scene once per process, cache baselines, run several seeds per process.
5. All fitness views of an individual in one `render_views` call; tune `batch_size`.
6. Fitness cache keyed on the (rounded) genome so elites/duplicates aren't re-rendered.
7. Later: render several individuals per call by zeroing opacities (gsplat 1.5.3 culls opacity
   < 1/255 and accepts `opacities` of shape `[B, N]`). Memory-hungry; FPS must still be measured
   on an actually pruned scene.
8. Optional multi-fidelity: score all on 4 views, re-score the top quarter on the full set.
9. Don't rotate fitness views per generation (noisy, elites not comparable).

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
