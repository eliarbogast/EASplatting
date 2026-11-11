# Running it

How to set up, run, check and measure everything, locally and on the UVM VACC. Commands run from the
repo root with the env active (`conda activate gsprune`). For the plan and design questions see
[`EA_PLAN.md`](EA_PLAN.md).

## 0. Check the machine first

```bash
python scripts/check_env.py            # safe anywhere, even before installing; paste the output when asking for help
python scripts/check_env.py --smoke    # after setup: also compile gsplat and render one Gaussian
```

| | Status |
|---|---|
| **Tested** | Pop!_OS 24.04 (Ubuntu-based, glibc 2.39), x86_64, RTX 5070 Ti (sm_120), driver CUDA 13.2, torch 2.8/cu128 |
| **Expected to work, not yet run** | other x86_64 glibc Linux (RHEL 9 on the VACC); GPUs sm_70 to sm_120 with driver CUDA >= 12.8. Checked: torch's wheel has kernels for all of them, and one gsplat build compiled for sm_70-sm_120 loads and renders on sm_120 (the other architectures are in the binary but were never executed) |
| **Fallback stack, not yet run** | torch 2.4.1/cu124 (picked automatically) for drivers CUDA 12.4-12.7, and for pre-Volta GPUs (sm < 7.0, e.g. GTX 10xx) |
| **Not supported** | Windows, macOS, aarch64 (untested), drivers older than CUDA 12.4 (override `CU`/`TORCH`/`TORCHVISION` at your own risk) |

Tools the scripts need: `git`, `sed`, `sort`, conda (or `python3 >= 3.9` with `venv`). Downloads and unzipping
use `scripts/download.py`, so `wget`, `curl` and `unzip` are not required. GPU memory: bonsai peaks at ~4.7 GB;
cards under 8 GB and outdoor scenes are untested (lower `batch_size` in `render_views` if you run out).

## 1. Environment (once)

```bash
bash scripts/setup_local.sh            # conda env "gsprune": torch + gsplat 1.5.3 + trainer deps, then a GPU smoke render
conda activate gsprune
export TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions
python -m pytest -q tests              # CPU tests with a fake renderer: all should pass
```

The script picks the torch build from your hardware: **torch 2.8 + cu128 by default** (kernels for sm_70 to
sm_120, incl. RTX 50xx Blackwell), torch 2.4.1/cu124 for older drivers or pre-Volta GPUs. There is no prebuilt
gsplat wheel for cu128, so gsplat JIT-compiles its CUDA kernels on first use (needs `nvcc`, which the script
installs into the conda env; ~35 s for one architecture, ~4 min for sm_70-sm_120; cached afterwards). It also
persists `CUDA_HOME`, `CPATH` and `LIBRARY_PATH` in the env. No root needed (Miniforge is fine).
Knobs (env vars): `CU`, `TORCH`, `TORCHVISION`, `GSPLAT`, `WITH_TRAINER=0` (skip gsplat's trainer),
`TORCH_CUDA_ARCH_LIST`, `NO_GPU_OK=1` (GPU-less login node), `PRINT_CONFIG=1` (show what it would do).

## 2. Data

```bash
bash scripts/get_data.sh               # default set below (~3-4 GB)
```

| What | Where | Why |
|---|---|---|
| Mip-NeRF 360 **bonsai** (images_2 + COLMAP) | `data/360_v2/bonsai` | main benchmark, small indoor scene |
| NerfBaselines 3DGS checkpoint for bonsai + its metrics json | `pretrained/nb/gaussian-splatting/mipnerf360/bonsai/` | baseline trained on the train split only; published test metrics to cross-check |
| Inria Tanks&Temples + Deep Blending inputs | `data/tandt`, `data/db` | small COLMAP scenes for training tests |
| Inria pretrained **truck** | `pretrained/inria/truck/point_cloud.ply` | quick render test (may have seen test views) |

More scenes: `bash scripts/get_data.sh mipnerf360 garden counter` and
`bash scripts/get_data.sh nb-checkpoints garden counter`. Mip-NeRF 360 scenes are range-extracted from the 12 GB
zip (flowers/treehill from the separate extra-scenes zip), so you only download what you ask for. Add
`FULLRES=1` if you'll train on Mip-NeRF 360 with gsplat (its parser re-downscales from full-res).

## 3. Render + metrics on a pretrained scene (the main check)

```bash
NB=pretrained/nb/gaussian-splatting/mipnerf360/bonsai
python -m easplatting.cameras data/360_v2/bonsai --images images_2 --scene $NB/point_cloud.ply   # frustum check
python -m easplatting.evaluate --scene $NB/point_cloud.ply --cameras data/360_v2/bonsai --images images_2 \
    --split test --nb-metrics $NB/bonsai.json --prune-opacity 0.005 --fps --cache-baselines \
    --out results/eval_bonsai_nb3dgs
```

Pass criteria (all in `results/eval_bonsai_nb3dgs/report.json`):
- `checks.frustum_median_visible` well above 5% (tens of %), and `checks.gt_size_matches_camera: true`.
- `checks.matches_nerfbaselines`: test PSNR / SSIM / LPIPS-VGG within 0.1 dB / 0.003 / 0.01 of NerfBaselines'
  published numbers. The strongest evidence that the whole chain (loader, cameras, renderer, metrics, protocol) is right.
- `compare/*.png` look right: GT | render | error x5. `prune_demo`: kept fraction and PSNR vs unpruned.

## 4. Train a splat yourself (data to splat)

```bash
bash scripts/train_gsplat.sh                       # truck, 7k steps: smoke test (~2 min on the 5070 Ti)
OUT=results/truck_gsplat_7k
python -m easplatting.evaluate --scene $OUT/ckpts/ckpt_6999_rank0.pt --cameras $OUT/cameras.json \
    --split test --lpips-net alex --gsplat-stats $OUT/stats/val_step6999.json --out results/eval_truck_gsplat_7k
```

`checks.matches_gsplat` should be true (our metrics vs the trainer's own eval on the same views). Then `STEPS=30000`
for a real baseline. A gsplat-trained scene lives in a normalized world frame: always pair it with `$OUT/cameras.json`,
never raw COLMAP.

## 5. Look at it: interactive viewer

```bash
python -m easplatting.viewer --scene $NB/point_cloud.ply --cameras data/360_v2/bonsai --images images_2
# open http://localhost:8080   (over SSH: ssh -L 8080:localhost:8080 <host>)
```

Sliders threshold opacity, max scale and max anisotropy (log10; the end marked "off" keeps everything). Live
readout of kept count, estimated size, render time and the current view's PSNR vs the unpruned render. View
modes: pruned, unpruned, removed only (what the thresholds delete), error x5. Buttons save the thresholds
(`results/viewer/thresholds.json`) or the pruned `.ply`; `--cameras` adds a jump-to-camera dropdown. It binds to
127.0.0.1 only (`--host 0.0.0.0` to expose it). The opacity-0.005 preset reproduces `evaluate`'s 87.9% kept on bonsai.

## 6. Measure this machine

```bash
python scripts/measure.py cost       # what does one EA fitness evaluation cost? scene size, GPU sharing, metrics (~2 min)
python scripts/measure.py metrics    # what do PSNR / SSIM / LPIPS mean for pruning? writes results/metrics_intuition/ (~3 min)
```

Re-run `cost` on every new GPU type and update the budget in [`EA_PLAN.md`](EA_PLAN.md). Keep each machine's
`report.json` too (it records GPU, torch, gsplat versions and timings).

## 7. Verified results (2026-11-02, RTX 5070 Ti, torch 2.8/cu128, gsplat 1.5.3)

| Run | Result |
|---|---|
| **Bonsai**, NerfBaselines 3DGS checkpoint, 1,247,650 Gaussians, 37 test views at 1559x1039 | PSNR 32.235 / SSIM 0.9421 / LPIPS-VGG 0.2622 vs NerfBaselines 32.197 / 0.9412 / 0.2538; all checks pass. Opacity > 0.005 keeps 87.9% at 60 dB vs unpruned. ~436 FPS single view, 4.6 GB peak |
| **Truck**, gsplat 7k steps, 2.05 M Gaussians, 2:16 training | PSNR 23.829 / SSIM 0.8527 / LPIPS-alex 0.1340 vs the trainer's own 23.827 / 0.8526 / 0.1340 (`matches_gsplat`) |

Raw reports and the fixes made during bring-up: [`archive/bring_up/`](archive/bring_up/README.md). Convention
worth remembering: LPIPS-VGG follows NerfBaselines (inputs rescaled to [-1,1]); the Inria repo feeds raw [0,1]
and reads ~0.05 lower (`--lpips-net vgg-raw`).

## 8. Running on the UVM VACC

Written before we have access, from the VACC docs ([cluster specs](https://www.uvm.edu/vacc/cluster-specs),
[partitions](https://www.uvm.edu/vacc/docs/beyond_basics/slurm_partitions/),
[submitting jobs](https://www.uvm.edu/vacc/docs/start_guide/running_jobs/)). Verify against `sinfo` /
`show_node_constraints` once we're on; the "Confirm with VACC" list below is what the docs don't say.

| GPU | Arch (`TORCH_CUDA_ARCH_LIST`) | Memory | Notes |
|---|---|---|---|
| H200 (NVL, SXM) | 9.0 | 141 GB | most GPUs; 4-8 per node |
| H100 | 9.0 | 80 GB | 2 nodes |
| A100 | 8.0 | 40 GB | 2 nodes; this is what `gpu-debug` gives |
| RTX Pro 6000 (Blackwell) | 12.0 | 96 GB | same architecture as Eli's desktop |

RHEL 9.6, Slurm, Lmod (CUDA 11/12/13 modules), shared GPFS storage. Partitions: `nvgpu` (GPU jobs, 48 h),
`gpu-debug` (3 h, 1 A100, max 2 submitted / 1 running per user), `gpu-preempt` (H200 nodes, **jobs can be
cancelled**); `general` has no GPUs. Pick GPUs with `--constraint`, e.g. `GPU_SKU:H200`, `GPU_MEM:80GB`, `gpu_any`.
The torch 2.8 + cu128 wheels carry sm_80, sm_90 and sm_120 kernels, so **one env and one gsplat build serve every node type**.

Checklist:
1. **Look before you build:** `python scripts/check_env.py` on the login node and again on a GPU node.
2. **Build the env on a login node** (no GPU needed), on project/GPFS storage, not `$HOME` (the conda CUDA toolkit
   alone is several GB): `NO_GPU_OK=1 TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" bash scripts/setup_local.sh`
3. **Pre-warm gsplat's JIT cache once** on a GPU node (a `gpu-debug` job is enough), with the same arch list and a
   shared cache in every job; otherwise a different list triggers a rebuild and, in an array, tasks race on the build lock:
   `export TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" TORCH_EXTENSIONS_DIR=/path/on/gpfs/torch_ext; python scripts/check_env.py --smoke-only`
4. **Make compute nodes self-sufficient** (they may be offline): fetch data on a login node
   (`bash scripts/get_data.sh mipnerf360 bonsai counter ...`, then `nb-checkpoints`) and the LPIPS weights
   (`TORCH_HOME=/path/on/gpfs/torch_home python scripts/check_env.py --prefetch-weights`), then export the same `TORCH_HOME` in jobs.
5. **Job preamble** (same in every job): activate the env, export `TORCH_EXTENSIONS_DIR`, `TORCH_CUDA_ARCH_LIST`,
   `TORCH_HOME`; request one GPU with a `--constraint`. Give each array task its own log/JSONL file (concurrent
   appends corrupt a shared file). Check `nvidia-smi` shows driver >= 570 (CUDA 12.8).
6. **One GPU SKU for every timed comparison**, e.g. `--constraint=GPU_SKU:H200`; re-run `measure.py cost` there and
   compare methods by evaluation count, not wall time. `gpu-debug` (A100) is for dry runs, not timings.
7. **Runs must be resumable** (skip finished seeds, write per-seed results as you go): required for `gpu-preempt`,
   good practice for `nvgpu`. Our runs are minutes long, so seed-level resume is enough.
8. **Parallelism = number of GPUs.** On the 5070 Ti, 1 / 2 / 4 processes sharing a GPU gave 75 / 70 / 76 evals/s in
   total: one process saturates it. Re-run `measure.py cost` on an H200 before assuming a bigger GPU behaves
   differently; until then spread seeds across GPUs (array tasks), not processes on one GPU.
9. **Logs:** progress meters should print plain periodic lines when not attached to a TTY; plots use matplotlib's `Agg`.
10. **Viewer on a node:** binds to 127.0.0.1 only; from a laptop `ssh -L 8080:<gpu-node>:8080 <vacc-login>`, then open `http://localhost:8080`.

Confirm with VACC (not stated in the docs we found): do compute nodes have internet (step 4 assumes not); exact GPU
request syntax (the docs' example uses only `--constraint`; is `--gres=gpu:1` also needed?); scratch/project paths,
quotas and purge policy (~2 GB per scene, ~1.4 GB per checkpoint); which conda / Python / compiler modules exist (we
assume none and install Miniforge ourselves); per-user limits on jobs and array tasks for `nvgpu`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| frustum check ~0% | gsplat-trained scene with raw COLMAP cameras: use the exported `cameras.json`; or wrong scene/cameras pair |
| PSNR < 15 dB, image looks shifted/garbage | wrong `--images` folder (intrinsics vs image size), or distorted (non-PINHOLE) COLMAP |
| PSNR a bit low, images look fine | background mismatch (NeRF-Synthetic: white), or `--rasterize-mode` vs how it was trained |
| GT size warning | `--images` doesn't match the resolution the scene was evaluated at |
| first render takes minutes | gsplat JIT compile (once); keep `TORCH_EXTENSIONS_DIR` set |
| "No CUDA GPUs are available" in a sandboxed shell | the sandbox hides `/dev/nvidia*`; run GPU commands outside it |
| LPIPS off vs a paper | different LPIPS convention/net; see section 7 (`--lpips-net vgg-raw` for Inria-style tables) |
| truck test PSNR >> 25 dB | Inria pretrained model saw the test views; use NB checkpoints for honest numbers |
