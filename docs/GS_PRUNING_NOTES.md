# Evolving Pruning Thresholds for 3D Gaussian Splatting — Developer Notes

Long reference doc (skim the section you need; start at `README.md`). Notes for the EA-based
3DGS pruning project (Eli & Areeb). Covers file
formats and conventions, metrics, fitness design, genome/operators, baselines,
experiment protocol, VACC usage, and known pitfalls. Pairs with `gs_loader.py`.

---

## 1. Project in one paragraph

Take a **pretrained** 3DGS scene, define a small vector of pruning thresholds
(min opacity, max scale, …), and evolve that vector with a GA (tournament
selection + crossover + mutation). Each candidate produces a keep-mask over the
Gaussians; the pruned scene is rendered from a set of cameras and compared to the
unpruned renders. Fitness rewards high quality and few Gaussians. Stretch goal:
evolve separate thresholds per spatial chunk of the scene.

The core loop is cheap — **mask → render N views → metrics** — so most of the
engineering is about doing that loop correctly and fast.

---

## 2. Scene file formats & conventions

### 2.1 3DGS `.ply` (Inria original / most exports)

Binary little-endian PLY, one `vertex` per Gaussian, all `float32`:

| Property | Count | Meaning | Stored as |
|---|---|---|---|
| `x, y, z` | 3 | center | world coords |
| `nx, ny, nz` | 3 | normals | **always 0, unused** |
| `f_dc_0..2` | 3 | degree-0 SH (base color) | raw SH coeff |
| `f_rest_0..44` | 45 | SH degrees 1–3 | raw SH, **channel-major** |
| `opacity` | 1 | opacity | **logit** |
| `scale_0..2` | 3 | per-axis size | **log** |
| `rot_0..3` | 4 | quaternion (w, x, y, z) | **unnormalized** |

- 62 floats → **248 bytes/Gaussian**. Real scenes: **~1–6 M Gaussians**, i.e.
  hundreds of MB (not "thousands" of Gaussians — worth fixing in the write-up).
- **Activations — always apply before thresholding:**
  - opacity: `sigmoid(opacity)` → [0, 1]
  - scale: `exp(scale)` → world units
  - rotation: `normalize(rot)`
  - base color: `rgb = 0.5 + 0.28209479 * f_dc` (SH C0 constant)
- `f_rest` layout: reshape to `(N, 3, 15)` then transpose to `(N, 15, 3)`.
  Sort property names **numerically** (`f_rest_10` after `f_rest_9`).
- SH dominates the file: `f_rest` is 45 of the 59 meaningful floats (~76%).

### 2.2 gsplat checkpoints (`.pt`)

`simple_trainer.py` saves `ckpt["splats"]` with keys
`means, quats, scales, opacities, sh0, shN` — **same log/logit conventions**,
already in `(N, K, 3)` SH layout.

### 2.3 Other formats you'll see

`.splat`, `.ksplat`, `.spz`, compressed PLY (SuperSplat) — viewer formats,
often lossy / SH-stripped. **Never use them as the baseline.**

### 2.4 Rule

Pick one canonical in-memory representation (`gs_loader.py` uses raw gsplat-style
tensors) and convert everything to it in one place. Most "it renders garbage"
bugs come from a missed activation or SH reshape.

---

## 3. Datasets & cameras

### 3.1 Which scenes

| Dataset | Scenes | Use |
|---|---|---|
| **Mip-NeRF 360** | 9 (bicycle, garden, stump, room, counter, kitchen, bonsai, + flowers, treehill) | Main benchmark; what LightGaussian & most pruning papers report |
| **Tanks & Temples** | truck, train | Standard secondary benchmark |
| **Deep Blending** | playroom, drjohnson | Standard secondary benchmark |
| **NeRF-Synthetic** | lego, chair, … | Small & fast → pipeline debugging only |

- **Decided (Nov 2026): NerfBaselines protocol on Mip-NeRF 360** — official
  `images_4` (outdoor) / `images_2` (indoor), llffhold-8 test split, LPIPS-VGG.
  Unpruned baselines = NerfBaselines `gaussian-splatting` checkpoints (trained on the
  train split only, with published per-scene metrics to cross-check against;
  reproduced 3DGS average 27.43 / 0.814 / 0.257 vs. paper 27.20 / – / 0.214).
  Get them with `scripts/get_data.sh` — see `docs/LOCAL_PIPELINE.md`.
- The Inria authors also released **pretrained models** for Mip-NeRF 360, T&T, and
  Deep Blending; these may have been trained on all images (including test views),
  so use them only for quick render tests.
- Suggested scope: develop on 1 small scene, run main experiments on 3–4
  Mip-NeRF 360 scenes (mix indoor/outdoor), optionally T&T truck.
- Outdoor/unbounded scenes behave very differently from indoor ones (huge
  background Gaussians) — include both.

### 3.2 Camera conventions (most common silent bug)

| Source | Stores | Convention |
|---|---|---|
| COLMAP `sparse/0/{cameras,images}.bin` | world→camera (quat + t) | OpenCV (x right, y down, z forward) |
| Blender `transforms.json` (NeRF-Synthetic) | camera→world 4×4 | OpenGL (y up, −z forward) |
| gsplat `rasterization()` wants | `viewmats` world→camera, `Ks` intrinsics | OpenCV |

**First sanity test:** render the unpruned scene from a training camera and
compare to the ground-truth photo. PSNR should be in the high 20s/30s dB. Black
or scrambled image = convention bug.

### 3.3 Splits & resolution

- Standard holdout: **every 8th image is test** (`llffhold = 8`).
- Mip-NeRF 360 standard: `images_4` for outdoor, `images_2` for indoor.
- NeRF-Synthetic: white (or black) background — `backgrounds=` in the
  rasterizer must match between baseline and pruned renders.

---

## 4. Metrics

All metrics on images in **[0, 1]**, float, clamped. Use the **same
implementation everywhere** (inside the EA and in final tables).

### 4.1 PSNR
- `PSNR = 10 · log10(1 / MSE)` (dB, higher = better).
- vs. ground-truth photos: 3DGS ≈ 27 dB on Mip-NeRF 360.
- vs. **unpruned renders** (our fitness reference): much higher (35–50+ dB), and
  **∞ for identical images** → clip (e.g. at 60 dB) or use MSE / ΔPSNR directly.
- Average over views as mean of per-view PSNR (state which — mean PSNR ≠ PSNR
  of mean MSE).

### 4.2 SSIM
- Structural similarity, ~[0, 1]. Standard: 11×11 Gaussian window, σ = 1.5.
- Implementations differ slightly (skimage, torchmetrics, pytorch-msssim, the
  3DGS repo's `ssim()`); set `data_range=1.0`. Compute on GPU.
- For mild pruning, values sit at 0.95–1.0 vs. the baseline → signal lives in
  tiny deltas. Consider `1 − SSIM` (DSSIM) in the fitness.

### 4.3 LPIPS (add this)
- Learned perceptual distance (lower = better). Every 3DGS paper reports
  **PSNR / SSIM / LPIPS**; you'll need it for comparison tables.
- State the backbone (VGG vs. AlexNet — papers differ; 3DGS uses VGG).
- Input convention matters too: the Inria repo feeds [0, 1] images to LPIPS without
  rescaling to [-1, 1] (`vgg-raw` in `metrics.py`; bonsai 0.207). NerfBaselines rescales
  properly (`vgg`, the default; 0.262 here vs 0.254 published), and `alex` matches both
  gsplat's trainer eval and NerfBaselines' `lpips`. Measured on bonsai, 2026-11-02. NerfBaselines notes that many papers evaluated on
  full-res images they downscaled themselves rather than the official downscaled set
  (worth ~0.5 dB PSNR); that likely explains part of the paper-vs-NerfBaselines LPIPS
  gap (0.214 vs 0.257) too, alongside the raw-vs-rescaled input difference above.
- Slower than PSNR/SSIM; fine for final evaluation, optional inside the EA.

### 4.4 Size & speed
- **Gaussian count** (and ratio to baseline). For pure pruning, bytes ∝ count.
  If you ever prune SH degree, report **bytes (MB)** separately.
- **FPS / render time**: not perfectly proportional to count (big Gaussians
  touch more tiles). If you claim speedups, measure properly:
  fixed resolution, warm-up iterations, `torch.cuda.synchronize()` around
  timing, median over many renders, same GPU type for all runs.

### 4.5 Two reference decisions — make them explicitly
1. **Quality reference.** Inside the EA, compare against **unpruned renders**
   (cheap, isolates pruning damage). In final tables, report **both**
   vs. unpruned and vs. ground-truth photos (the latter is what papers report).
2. **Which views.** The EA will overfit to the views it's scored on. **Evolve on
   a subset of training views, report on held-out test views.**

---

## 5. Fitness function

### 5.1 The problem with the proposal's formula

`fitness = quality − (count / baseline_count)`

The two terms are on different scales:
- **PSNR** (~35–50) swamps the size term in [0, 1] → EA keeps nearly everything.
- **SSIM** (~0.97–1.0) is swamped by the size term → EA prunes aggressively
  regardless of quality.

Either way, the result is set by the choice of metric, not by the trade-off.

### 5.2 Better options (pick one as primary, keep others as ablations)

**A. Weighted, normalized scalar**

```
fitness = − ΔPSNR − λ · (count / baseline_count)
ΔPSNR   = PSNR_cap − PSNR(pruned vs. unpruned)      # ≥ 0, quality loss
```
Sweep λ (e.g. 0.5, 1, 2, 5) and report the resulting points. Simple, fits the
current GA directly.

**B. Constrained**

```
maximize   1 − count / baseline_count
subject to ΔPSNR ≤ τ   (e.g. τ = 0.5 dB)     # or SSIM ≥ 0.99, LPIPS ≤ ε
penalty:   fitness −= M · max(0, ΔPSNR − τ)
```
Most interpretable claim: "X% fewer Gaussians at < 0.5 dB loss."

**C. Multi-objective (recommended as the interesting EA contribution)**
- NSGA-II (or MAP-Elites with size as a behavior descriptor) on
  (quality loss, size ratio) → a **Pareto front**, no arbitrary λ.
- Other methods become single points you can plot against the front.
- `pymoo` or `DEAP` provide NSGA-II out of the box.

### 5.3 Fitness hygiene
- Combine metrics only after normalizing (ΔPSNR in dB, DSSIM, LPIPS each scaled).
- Guard degenerate genomes: if the mask keeps 0 Gaussians (or < 1%), return a
  large penalty without rendering.
- Fitness is deterministic for a given genome + view set — cache it
  (`dict[tuple(genome_rounded)] → fitness`) to avoid re-rendering duplicates.
- If you subsample views per generation (for speed), fitness becomes noisy:
  re-evaluate elites, or use a fixed view subset.

---

## 6. Genome & operators

### 6.1 Candidate genes

| Gene | Encoded as | Prunes Gaussians where… |
|---|---|---|
| min opacity | logit or log10 | `sigmoid(opacity) < t` |
| max scale | log | `max(exp(scale)) > t` |
| min scale / min volume | log | `prod(exp(scale)) < t` (microscopic) |
| max anisotropy | log | `max/min scale > t` (needle-like) |
| importance threshold | percentile | LightGaussian-style score below the p-th percentile |
| SH degree cap (optional) | int 0–3 | drop `f_rest` above degree d (big byte savings) |

- **Encode in transformed space** (log / logit): thresholds span orders of
  magnitude; linear encoding wastes most of the search range.
- Use `python gs_loader.py scene.ply` to read the percentile distributions and
  set gene bounds from them (e.g. min-opacity range ≈ [1e-4, 0.2]).
- **Max-scale pruning is dangerous** in unbounded scenes: sky/background are
  represented by huge Gaussians — removing them punches holes. This is actually
  a good argument for scene-specific tuning (cite it as motivation).
- Importance score (LightGaussian): roughly `Σ_views hits × opacity × volume^β`.
  It needs one pass over training views to count per-Gaussian hits
  (gsplat's rasterization `info` gives per-Gaussian radii / visibility).
  Compute once per scene and cache.

### 6.2 Operators (real-valued genome)
- Selection: tournament (k = 2–4) as planned, plus **elitism** (keep top 1–2).
- Crossover: BLX-α or SBX (simulated binary crossover); uniform crossover is
  weak for few real genes.
- Mutation: Gaussian, σ ≈ 5–10% of gene range, clip to bounds.
- Population/generations: with 3–6 genes, ~20–40 individuals × 30–100
  generations is plenty. Budget = pop × gens evaluations — log it.

### 6.3 Alternative genome (worth a sentence in the write-up)
A direct per-Gaussian binary mask (millions of bits) is the "fully dynamic"
version but is intractable for a GA; the threshold vector is a compact
**indirect encoding**. Spatial chunks (stretch goal) sit in between.

---

## 7. Baselines (needed for any significance claim)

1. **Unpruned** scene (upper bound on quality, lower bound on compression).
2. **Fixed hand-tuned thresholds** — e.g. 3DGS's own opacity cull (0.005).
3. **LightGaussian-style global pruning** — prune the bottom p% by importance
   score, for p ∈ {50, 60, 66, 75}, **without** fine-tuning.
4. **Random search with the same evaluation budget** as the GA.
5. (Optional) **CMA-ES** with the same budget — a strong continuous optimizer;
   good to show whether the GA's operators actually matter.

The search space is low-dimensional and fairly monotone (lower thresholds keep
more), so random search / CMA-ES may match the GA. That's a legitimate finding —
but you need the baseline to know.

---

## 8. Experiment protocol

- **Seeds:** 20–30 independent runs per (scene × method × setting).
- **Views:** evolve on a fixed subset of training views (e.g. 8–16, possibly at
  reduced resolution); report on all held-out test views at full eval resolution.
- **Statistics:** Mann–Whitney U (unpaired) or Wilcoxon signed-rank (paired by
  seed/scene) on final quality at matched size (or size at matched quality);
  report median + IQR; Holm/Bonferroni if many comparisons.
- **Plots:**
  - Best-fitness-vs-evaluations curves (median ± IQR across seeds) for GA vs.
    random search vs. CMA-ES.
  - Quality-vs-size scatter / Pareto front with baselines as points.
  - Side-by-side renders + error heatmaps (`|pruned − unpruned|`) on a few views.
  - Histogram of opacity/scale with the evolved thresholds overlaid.
- **Fine-tuning:** nearly all prior pruning methods fine-tune after pruning to
  recover quality. Prune-only keeps the EA loop cheap. Suggested: evolve
  prune-only, then optionally fine-tune the **final** solution for a few thousand
  iterations and report both. Be explicit about which numbers are which.
- **Logging (from day one):** one JSONL/CSV row per evaluation — run id, seed,
  scene, generation, genome, count, PSNR/SSIM(/LPIPS), wall time. Save the best
  genome per generation. Plotting later is then trivial.
- **Config files:** every run driven by a YAML/JSON config (scene, method,
  λ/τ, pop, gens, seed). Same config → same experiment, whether on a laptop or
  the VACC (addresses the "testing the same thing" goal in Week 4).

---

## 9. Compute & the VACC

### 9.1 Don't use a SLURM task per individual
One fitness eval with the scene already on the GPU is well under a second.
A separate job per individual spends nearly all its time on queueing, Python/CUDA
startup, and loading a ~500 MB scene.

**Instead:**
- **One process = one full EA run.** Load scene once, keep it on GPU, evaluate
  the whole population in-process.
- **SLURM array over (scene × method × seed × setting)**, e.g. `--array=0-119`,
  index → config.
- **Cache baseline renders** per scene once (tensors in `.pt`/`.npy`); never
  re-render the unpruned scene during evolution.

### 9.2 Practicalities
- gsplat JIT-compiles CUDA kernels on first use. Either install a prebuilt
  wheel matching the cluster's CUDA/PyTorch, or set a persistent
  `TORCH_EXTENSIONS_DIR` and compile once before launching arrays (otherwise 100
  array tasks all compile at once, or collide on the same cache).
- Check which GPU types the partition gives you; pin `--gres=gpu:<type>:1` if
  you report FPS, so timings are comparable.
- Memory: a 6 M-Gaussian scene is ~1.5 GB of float32 on GPU; rendering 1080p
  views adds more. Small GPUs may need reduced resolution for fitness.
- Write outputs per task to separate files (`results/{run_id}.jsonl`) — no
  shared-file appends from many tasks.
- Do short local smoke tests (2 gens × 4 individuals) before every big array.

---

## 10. Stretch goal: spatial chunks

- Assign each Gaussian to a chunk by its center: uniform voxel grid (k³ cells),
  k-means on positions, or an octree (adaptive to density).
- Genome becomes `k × p` genes → use **cooperative coevolution** (evolve each
  chunk's genes as a sub-population, evaluate in context of the others' best)
  or a CMA-ES over the whole vector.
- Watch for:
  - **Seams** at chunk boundaries (visible discontinuities in density).
  - Chunks seen by few training views → weak/noisy fitness signal; consider
    tying their parameters to the global ones.
  - Fitness attribution: a global image metric doesn't say which chunk caused
    the damage. Per-chunk contribution (render with only that chunk's mask
    changed) costs more renders.
- Start with k = 2–4 per axis (8–64 chunks) before going finer.

---

## 11. Notes for the write-up / related work

- **LightGaussian is not gradient search over thresholds.** Its pruning is a
  global **importance-score percentile cut** followed by **fine-tuning**; the
  headline 15× comes mostly from **SH distillation + vector quantization**, not
  pruning alone. Reframe the related work accordingly, and compare against its
  **pruning stage**, not its 15× number.
- The "stuck in local optima" argument needs care: a percentile threshold isn't
  optimized by gradient at all. A sharper framing: prior methods use **fixed,
  scene-agnostic** thresholds/percentiles chosen by hand; you search them **per
  scene** (and per region in the stretch goal).
- "Thousands of Gaussians" → "millions".
- Citation name: text says "Hahlbom", reference says "Hahlbohm". Double-check
  that Faster-GS's analysis actually covers pruning thresholds (vs. general
  training-speed improvements) before leaning on it in the rationale.
- "PyTorch's GS implementation" — there's no official one; the reference
  implementation is Inria's `diff-gaussian-rasterization` (CUDA, PyTorch
  bindings). Simplest to commit to **gsplat**.
- Brute-force claim: with 3–6 genes, a coarse grid search *is* feasible on one
  scene — consider running one as a sanity-check landscape plot rather than
  claiming it's infeasible.
- **Closest prior work — cite and position against:**
  - *LP-3DGS* (NeurIPS 2024): learns a per-scene pruning ratio with a trainable
    Gumbel-sigmoid mask over an importance score (RadSplat or Mini-Splatting score);
    ~60–70% pruned on Mip-NeRF 360 at ≈0 dB loss. Gradient-based, needs training.
  - *Speedy-Splat*: soft + hard pruning during training (PUP-3DGS-style score);
    10.6× fewer Gaussians, 6.7× FPS, −0.6 dB on Mip-NeRF 360.
  - *PUP 3D-GS*, *Mini-Splatting*, *LightGaussian*, *EAGLES*, *Compact3D*, *ELMGS*.
  - All prune **during training and fine-tune**. Our angle: **post-hoc, training-free,
    gradient-free** threshold search; compare against their pruning stage only.
  - A quick search found no prior EA/GA work on 3DGS pruning thresholds (not exhaustive).
- *Faster-GS* (Hahlbohm et al., CVPR 2026) consolidates training optimizations (up to
  5× faster training); its summary doesn't mention pruning thresholds — read it before
  citing it as precedent for threshold tuning.
- Renderer choice: gsplat (batched Python API, maintained, matches Inria quality, less
  memory). Inria's `diff-gaussian-rasterization` is what most pruning papers build on —
  only needed if we fine-tune with their recipe. Brush (Rust/WebGPU) runs on Macs but has
  no Python API for the inner loop.

---

## 12. Pitfall checklist

- [ ] Thresholds applied to **activated** values (sigmoid opacity, exp scale)
- [ ] SH `f_rest` reshaped channel-major → `(N, K, 3)`
- [ ] Camera convention verified by rendering unpruned vs. GT photo
- [ ] Same background color for baseline and pruned renders
- [ ] Images clamped to [0, 1] before metrics; same SSIM/LPIPS implementation everywhere
- [ ] PSNR vs. unpruned clipped (no ∞)
- [ ] Fitness terms on comparable scales (or multi-objective)
- [ ] Evolve on train views, report on held-out test views
- [ ] Empty/near-empty masks short-circuited
- [ ] Baseline renders cached; scene loaded once per run
- [ ] Random search (same budget) baseline included
- [ ] 20–30 seeds per condition; non-parametric tests
- [ ] Every run reproducible from a config + seed; results logged per evaluation
- [ ] FPS measured with synchronize + warm-up on a fixed GPU type

---

## 13. Code layout (repo: eliarbogast/EASplatting)

```
EASplatting/
├── gs_loader.py          # load/activate/prune/save (done)
├── cameras.py            # COLMAP / cameras.json / transforms.json → viewmats, Ks, splits (done)
├── render.py             # gsplat wrapper, baseline caching, contributions, FPS (done)
├── metrics.py            # PSNR / SSIM / LPIPS (done; one implementation each)
├── evaluate.py           # scene + cameras → renders → metrics → report.json (done)
├── export_gsplat_cameras.py  # cameras in gsplat-trainer's normalized frame (done)
├── viewer.py             # interactive threshold-pruning viewer (done)
├── scripts/              # setup_local.sh, get_data.sh, train_gsplat.sh, fetch_zip_members.py
├── tests/                # CPU tests with a fake gsplat
├── genome.py             # gene bounds, encode/decode, genome → keep-mask
├── fitness.py            # mask → renders → metrics → fitness (A/B/C variants)
├── algorithms/
│   ├── ga.py             # tournament + SBX/BLX + Gaussian mutation + elitism
│   ├── random_search.py
│   └── cmaes.py          # optional (pip install cma)
├── run.py                # one EA run from a config; writes results/{run_id}.jsonl
├── configs/              # YAML per experiment
├── slurm/array.sbatch    # array over configs × seeds
└── analysis/             # plots + stats notebooks
```

Build order: `gs_loader` → `cameras` + `render` (verify unpruned vs. GT) →
`metrics` (verify against a known value, e.g. 3DGS paper numbers on one scene) →
`genome` + `fitness` → random search (simplest optimizer, validates the loop) →
GA → VACC arrays.
