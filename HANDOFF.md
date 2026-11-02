# Handoff (2026-11-02)

Short status for the next session, human or Claude. Front door is `README.md`.

## Status

The local pipeline is brought up and verified on Eli's machine (RTX 5070 Ti, torch 2.8 + cu128,
gsplat 1.5.3): `gs_loader`, `cameras`, `render`, `metrics`, `evaluate`, `export_gsplat_cameras`,
`viewer`, and the `scripts/` all run for real; 28 CPU tests pass. Results and the fixes made along
the way are in [`docs/bring_up/README.md`](docs/bring_up/README.md). The gate for EA work
(a real gsplat run passing `evaluate.py`'s checks) is met.

**Not started:** `genome.py`, `fitness.py`, search algorithms, VACC scripts.
**Not run:** `get_data.sh inria-pretrained` (Inria's released models; not needed for the gate).

## Decisions made (and why)

- **Renderer: gsplat 1.5.3** (latest release; 1.6.0 on main is unreleased with a different trainer
  API). Batched multi-camera forward renders from Python, masks = tensor indexing.
- **Benchmark: NerfBaselines Mip-NeRF 360 protocol**: official `images_4`/`images_2`, llffhold-8
  test split, LPIPS-VGG on [-1,1]-rescaled inputs (this matches NB; the raw-[0,1] Inria convention
  reads ~0.05 lower and is kept as `vgg-raw`). Unpruned baseline = NerfBaselines'
  `gaussian-splatting` checkpoints (trained on the train split only, published per-scene metrics).
  Inria's own pretrained models may include test views, so use them only for quick tests.
- **Dev scene: bonsai** (indoor, `images_2`, 1.25 M Gaussians; outdoor scenes run to several million).
- **Tanks & Temples:** Inria's `tandt_db` (not NerfBaselines' T&T, which is preprocessed differently).
- **Fitness reference:** unpruned renders (cached by `evaluate.py --cache-baselines`), not GT photos;
  final tables report both.

## Next steps

1. Settle the open questions in [`docs/EA_PLAN.md`](docs/EA_PLAN.md) (fitness form, genes, views, scenes).
2. `genome.py`: gene bounds in transformed space (log scale, logit/log opacity); decode -> mask.
   `viewer.keep_mask` / `gene_stats` already implement opacity, max-scale and anisotropy thresholds.
3. `fitness.py`: mask -> render fitness views (train subset) -> dPSNR vs cached baselines
   (`metrics.psnr_fast`) + size term; short-circuit empty masks. Cost/budget tricks: `docs/EA_PLAN.md`.
4. Random search with the same evaluation budget first (validates the loop), then the GA
   (tournament + SBX/BLX + Gaussian mutation + elitism).
5. VACC: one process per (scene x method), seeds sequentially; SLURM array over those; persistent
   `TORCH_EXTENSIONS_DIR`. Checklist and VACC hardware: [`docs/CLUSTER.md`](docs/CLUSTER.md).
