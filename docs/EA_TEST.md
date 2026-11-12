# EA test run (branch `ea-test`)

A first end-to-end run of the evolutionary search on bonsai, built on Areeb's draft. It checks that the loop
works; it is not a result.

```bash
python -m easplatting.run_ea      # needs the bonsai data (scripts/get_data.sh) and a GPU; takes ~4 s
```

## What it does
- **Scene:** bonsai, 1,247,650 Gaussians (NerfBaselines 3DGS checkpoint).
- **Genome:** 3 numbers in [0, 1] = min opacity, max scale, max anisotropy, each mapped onto that gene's log10
  range for the scene (`genome.py`); the outer 5% of a range means the gene is off.
- **Fitness:** prune with the decoded thresholds, render 8 train views at half resolution, PSNR vs the
  unpruned render of the same views. If PSNR >= 40 dB the genome is feasible and scores `1 - kept fraction`;
  otherwise it scores below every feasible genome (`fitness.py`).
- **EA:** 16 parents, 32 children, 5 generations, mutation size 0.1, tournament of 4 with 2 winners, elitism,
  no crossover (`ea.py`).

## Result (seed 0)

| Generation | Gaussians kept | PSNR vs unpruned | Diversity |
|---|---|---|---|
| 0 | 77.2% | 47.3 dB | 0.232 |
| 1 | 75.0% | 43.4 dB | 0.214 |
| 2 | 74.8% | 43.3 dB | 0.093 |
| 3 | 72.1% | 40.7 dB | 0.036 |
| 4 | **71.2%** | **40.3 dB** | 0.050 |

- The best genome (min opacity 0.046, max scale 2.2, anisotropy off) is feasible and sits right on the
  40 dB constraint, as expected: it removes as much as the constraint allows.
- Rendering the same genome on the same 8 views at **full resolution** gives **40.3 dB** too, so the half-res
  fitness agreed with full-res here.
- Consistent with the earlier larger prototype (pop 100, 3 seeds): ~70.5% kept at 40 dB within ~1,000
  evaluations, and 2-5 points better than random search with the same budget.
- Caveat: one seed, 5 generations, a population of 16. It shows the plumbing works, not how well the GA does.
  Diversity dropping from 0.23 to 0.04 in four generations says this population is small for a real run.
- For scale: 40 dB vs the unpruned render is a real loss of roughly 0.8 dB PSNR vs the photos on bonsai
  ([`EA_PLAN.md`](EA_PLAN.md), "How to measure fitness").

## Rendering low-res views for the fitness
```python
train_views = cams.subset(select_indices(len(cams), "train", k=8))   # 8 evenly spaced train views
fitness_views = train_views.scaled(0.5)                               # intrinsics and image size x 0.5
baselines = render_views(scene, fitness_views)                        # unpruned scene, same low res
rendered = render_views(prune(scene, mask), fitness_views)            # pruned scene, same low res
psnr = psnr_fast(rendered, baselines, max_db=60).mean()               # pruned vs unpruned, never vs photos
```
- `Cameras.scaled(s)` scales the focal lengths, principal point and image size, so gsplat renders natively at
  the smaller size (no resizing of an image). The views are fixed for the whole run.
- Both sides use the same resolution, so resolution effects cancel; what remains is the damage from removed
  Gaussians, which shows up as patches many pixels wide and survives halving the image.
- **Why we trust it** (bonsai, 80 genomes, half-res vs full-res): same ranking (Spearman 0.997), PSNR differs by
  -0.01 dB on average (std 0.32, about +-0.1 dB near 40 dB), and the 40 dB feasible/infeasible call agrees for
  98%. Cost: ~12 ms per evaluation vs ~0.5 s for all 255 train views at full resolution.
- **Guard rails for the real experiment:** check the ranking on each new scene (Spearman >= 0.95 over ~50 random
  genomes); re-evaluate final solutions at full resolution on the held-out test views and report those numbers;
  keep a small margin above tau; genes that remove tiny Gaussians (a min-scale gene) need a higher resolution
  because half-res under-sees fine detail. Only validated on bonsai so far.
