# EA design: open questions and feedback ideas

For Eli + Areeb to settle as we write the EA. "Lean" = my current suggestion, not a decision.
Background: `docs/GS_PRUNING_NOTES.md` §5–8, `HANDOFF.md` "Compute plan". Already decided (not
repeated here): unpruned renders as the in-loop reference, evolve on train views / report on test,
PSNR-only in the loop, NerfBaselines protocol.

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
   ρ ≳ 0.95 vs full fitness (HANDOFF item 2) before trusting it.
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
15. **Seeds / pop / gens:** 20-30 seeds; pop 32 x 50 gens is plenty per HANDOFF. Any reason for more?

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
- **viser viewer** (installed with the trainer deps): load the scene, move threshold sliders, watch
  Gaussians disappear and the count/FPS update. Doubles as a sanity check of the mask logic.

*Lean for a first pass:* JSONL logging + terminal line + `progress.png` + TensorBoard, then Pareto
and convergence plots from the JSONL; landscape slice and removed-Gaussians render for the write-up.
