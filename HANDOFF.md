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
2. `genome.py`: gene bounds in transformed space (log scale, logit/log opacity); decode → mask.
   Use `python gs_loader.py <ply>` percentiles to set bounds.
3. `fitness.py`: mask → render fitness views (train split subset) → ΔPSNR vs cached baselines
   (`metrics.psnr_fast`) + size term. Normalize terms (see notes §5); short-circuit empty masks.
4. Random search with the same evaluation budget first (validates the loop), then the GA
   (tournament + SBX/BLX + Gaussian mutation + elitism).
5. VACC: one process per EA run, SLURM array over (scene × method × seed); persistent
   `TORCH_EXTENSIONS_DIR`.

## Open questions for Eli / Areeb

- Fitness form: weighted (λ sweep), constrained (ΔPSNR ≤ τ), or multi-objective (NSGA-II)?
- Which scenes for the main experiments (suggest 3–4 Mip-NeRF 360, indoor + outdoor)?
- Fine-tune after pruning, or strictly training-free? (Positioning vs LP-3DGS / Speedy-Splat.)
