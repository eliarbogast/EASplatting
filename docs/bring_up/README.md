# Bring-up reports (2026-11-02)

First real-GPU runs: RTX 5070 Ti, torch 2.8.0+cu128, gsplat 1.5.3. Copies of the git-ignored
`results/eval_*/` outputs (small files only; no renders).

| File | Run | Result |
|---|---|---|
| `bonsai_nb3dgs_*` | NerfBaselines 3DGS checkpoint, bonsai, test split, `images_2` | PSNR 32.235 / SSIM 0.9421 / LPIPS-VGG 0.2622 vs NB 32.197 / 0.9412 / 0.2538; all checks pass |
| `truck_gsplat7k_*` | gsplat 7k-step truck, test split, LPIPS-alex | matches trainer's own eval (23.829 dB vs 23.827) |

Commands: see `docs/LOCAL_PIPELINE.md` steps 2–3.
