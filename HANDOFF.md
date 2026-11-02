# Handoff — pipeline setup (branch `pipeline-setup`, 2026-11-02)

Context for the next session (human or Claude). Written at the end of the planning chat
that produced this code.

## Status

| Piece | State |
|---|---|
| `gs_loader.py` | done; round-trip tested; PLY layout matches gsplat's own loader/exporter |
| `cameras.py` | done; COLMAP bin/txt, Inria + exported `cameras.json`, Blender; tested on synthetic scenes |
| `render.py` | done; tested with fake rasterizer (batching, empty scene, cache, contributions) |
| `metrics.py` | done; torchmetrics PSNR/SSIM tested; **LPIPS untested** (weights couldn't download in the cloud) |
| `evaluate.py` | done; end-to-end test with fake rasterizer on CPU |
| `export_gsplat_cameras.py` | written against gsplat v1.5.3 `Parser`; **untested** (needs pycolmap + data) |
| `scripts/fetch_zip_members.py` | done; tested on local zips incl. zip64 |
| `scripts/get_data.sh`, `setup_local.sh`, `train_gsplat.sh` | written; syntax-checked; **never run** (cloud had no GPU / no access to data hosts) |
| fitness / genome / EA | not started |

**Nothing has run on a real GPU yet.** First job: `docs/LOCAL_PIPELINE.md` steps 0–3 on the
Linux + NVIDIA machine, fixing whatever breaks.

## Decisions made (and why)

- **Renderer: gsplat 1.5.3** (latest release; 1.6.0 on main is unreleased with a different
  trainer API). Batched multi-camera forward renders from Python, masks = tensor indexing.
  Inria's `diff-gaussian-rasterization` only needed if we add post-pruning fine-tuning with
  their recipe.
- **Benchmark: NerfBaselines Mip-NeRF 360 protocol** — official `images_4`/`images_2`,
  llffhold-8 test split, LPIPS-VGG with [0,1] inputs. Unpruned baseline = NerfBaselines'
  `gaussian-splatting` checkpoints (trained on train split only; published per-scene
  metrics). Inria's own pretrained models may include test views → only for quick tests.
- **Local dev scene: bonsai** (indoor, images_2; far fewer Gaussians than the outdoor
  scenes like garden/bicycle, which run to several million).
- **Tanks & Temples:** Inria's `tandt_db` (not NerfBaselines' T&T, which is preprocessed
  differently).
- **Fitness reference:** unpruned renders (cached by `evaluate.py --cache-baselines`), not
  GT photos; final tables report both.

## Unverified assumptions (check on first real run)

1. NerfBaselines checkpoint URL pattern
   `huggingface.co/nerfbaselines/nerfbaselines/resolve/main/gaussian-splatting/mipnerf360/<scene>.zip`
   (+ `.json` metrics) and that the zip contains `point_cloud/iteration_*/point_cloud.ply`
   — taken from NerfBaselines' source, not downloaded. `get_data.sh` searches the zip for the
   PLY and fails loudly if missing.
2. NerfBaselines metrics json keys (`psnr`, `ssim`, `lpips_vgg`, top level or under
   `metrics`, scalars or per-image lists) — `evaluate.crosscheck` is lenient; inspect the file.
3. Mip-NeRF 360 zips have top-level `<scene>/` folders (true for 360_v2.zip per NerfBaselines'
   code; extra-scenes zip assumed the same).
4. Prebuilt gsplat wheel exists for torch 2.4 + cu124 (else PyPI JIT build; needs nvcc).
5. gsplat v1.5.3 `examples/requirements.txt` builds cleanly (fused-ssim, fused-bilagrid need nvcc).

## Next steps

1. Local bring-up (`docs/LOCAL_PIPELINE.md` 0–3). Commit fixes. Save `report.json`s.

> **Gate (Eli, 2026-11-02): don't start steps 2+ (genome / fitness / EA) until step 1 has
> produced a successful real gsplat run on Eli's Linux + NVIDIA machine** — i.e. the
> bonsai `report.json` passes its checks (frustum, GT size, matches NerfBaselines). Until
> then, work only on bring-up fixes, tests and docs. The plan below is context, not a to-do.

2. `genome.py`: gene bounds in transformed space (log scale, logit/log opacity); decode → mask.
   Use `python gs_loader.py <ply>` percentiles to set bounds.
3. `fitness.py`: mask → render fitness views (train split subset) → ΔPSNR vs cached baselines
   (`metrics.psnr_fast`) + size term. Normalize terms (see notes §5); short-circuit empty masks.
   Expose fitness view count and render scale as settings (see "Compute plan" below).
4. Random search with the same evaluation budget first (validates the loop), then the GA
   (tournament + SBX/BLX + Gaussian mutation + elitism).
5. VACC: one process per (scene × method), running seeds sequentially; SLURM array over
   those; persistent `TORCH_EXTENSIONS_DIR`.

## Compute plan (context for steps 2–5; decided 2026-11-02)

**Datasets by purpose**
- Dev/debug of the EA loop: NeRF-Synthetic (e.g. lego; a few 100k Gaussians, small images)
  or a 7k-step gsplat truck — whole EA runs in seconds. Not for results.
- Main results: Mip-NeRF 360, 3–4 scenes, indoor first (bonsai, counter), then garden /
  bicycle. This keeps comparability with LP-3DGS, Speedy-Splat, PUP-3DGS, LightGaussian.
- Optional secondary table: all 8 NeRF-Synthetic scenes (cheap; LP-3DGS reports it).
  Don't make it the main benchmark — no unbounded background, which is where scene-specific
  thresholds should matter most.

**Budget estimate** (replace with measured `timing.render_s_per_view` from `report.json`)
- 3DGS on Mip-NeRF 360 ≈ 134 FPS avg at eval resolution (Speedy-Splat paper) ≈ 7.5 ms/view.
- Fitness = ~8 train views at reduced resolution ≈ 20–60 ms → 15–50 evals/s; faster as
  individuals prune more.
- Run = pop 32 × 50 gens = 1,600 evals ≈ 0.5–2 min (3–6 genes don't need more).
- Study = 4 scenes × 4 methods × 30 seeds = 480 runs ≈ 4–16 GPU-hours.

**Making evaluations cheap (by expected impact)**
1. Cheap fitness, full final eval: fixed 8–16 TRAIN views at reduced resolution during
   evolution; only final/best genomes evaluated at protocol resolution on TEST views.
2. Validate the shortcut: ~50 random genomes, cheap vs full fitness (all train views, full
   res); require Spearman ρ ≳ 0.95. Report it (one plot).
3. PSNR/MSE only in the loop (`metrics.psnr_fast` vs cached baselines). SSIM/LPIPS at final
   eval only (`fused-ssim` is available if SSIM is wanted in-loop).
4. Scene loaded + activated once per process; baselines cached; run several seeds per
   process. Masks are one comparison per Gaussian — negligible.
5. All fitness views of an individual in one `render_views` call; tune `batch_size`.
6. Fitness cache keyed on (rounded) genome — elites/duplicates aren't re-rendered.
7. Advanced, after the basic loop works: render several individuals per call. gsplat 1.5.3
   culls Gaussians with opacity < 1/255 at projection (checked in
   `ProjectionEWA3DGSFused.cu`) and `opacities` accepts a batch dim `[B, N]`, so pruning can
   be encoded as zeroed opacities for B individuals × C cameras. Memory-hungry; most useful
   on small scenes. FPS must still be measured on an actually pruned scene.
8. Optional multi-fidelity: score everyone on 4 views, re-score the top quarter on the full
   fitness set — only if evals dominate.
9. Don't rotate fitness views per generation (noisy, elites not comparable); guard against
   overfitting with held-out test views instead.

## Open questions for Eli / Areeb

- Fitness form: weighted (λ sweep), constrained (ΔPSNR ≤ τ), or multi-objective (NSGA-II)?
- Which scenes for the main experiments (suggest 3–4 Mip-NeRF 360, indoor + outdoor)?
- Fine-tune after pruning, or strictly training-free? (Positioning vs LP-3DGS / Speedy-Splat.)
