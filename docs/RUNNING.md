# Running it

How to set up, run, check and measure everything, on this machine. Commands run from the
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
| **Expected to work, not yet run** | other x86_64 glibc Linux; GPUs sm_70 to sm_120 with driver CUDA >= 12.8. Checked: torch's wheel has kernels for all of them, and one gsplat build compiled for sm_70-sm_120 loads and renders on sm_120 (the other architectures are in the binary but were never executed) |
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

The same CPU tests run in GitHub Actions on every push (`.github/workflows/tests.yml`: CPU torch, fake gsplat from `tests/`, then `pip install -r requirements.txt`).
A new third-party import in `easplatting/` or `tests/` must also go into `requirements.txt`, or CI fails with `ModuleNotFoundError` while your machine passes.

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

## 7. Reference numbers (bring-up, 2026-11-02; RTX 5070 Ti, torch 2.8/cu128, gsplat 1.5.3)

Both pipelines match their references (bonsai vs NerfBaselines: 32.235 dB vs 32.197; truck vs gsplat's own eval: 23.829 vs 23.827). The full table,
raw reports and the fixes made during bring-up are archived in [`archive/bring_up/`](archive/bring_up/README.md). Convention worth remembering:
LPIPS-VGG follows NerfBaselines (inputs rescaled to [-1,1]); the Inria repo feeds raw [0,1] and reads ~0.05 lower (`--lpips-net vgg-raw`).

## 8. Running a GPU job from Claude Code (pop-os)

Claude's sandbox hides `/dev/nvidia*`, so plain commands see no GPU ("No CUDA GPUs are available"). Use the root-owned
helper `/usr/local/bin/gpu-run` (listed in `sandbox.excludedCommands`; network off, home read-only, the repo writable).
Run it from the repo root as ONE plain command: a leading `cd ... &&`, a `source`, or a trailing `| tail` makes it
compound, it then runs inside the sandbox, and fails with "bwrap: No permissions to create new namespace".

```bash
/usr/local/bin/gpu-run -- /usr/bin/env CUDA_HOME=/home/eli/miniforge3/envs/gsprune PATH=/home/eli/miniforge3/envs/gsprune/bin:/usr/bin:/bin CPATH=/home/eli/miniforge3/envs/gsprune/targets/x86_64-linux/include LIBRARY_PATH=/home/eli/miniforge3/envs/gsprune/targets/x86_64-linux/lib TORCH_CUDA_ARCH_LIST=12.0 /home/eli/miniforge3/envs/gsprune/bin/python -m easplatting.experiment --scene bonsai
```

The `env` variables replace `conda activate` (they are what `~/miniforge3/envs/gsprune/conda-meta/state` persists).
Without `CUDA_HOME`, gsplat prints "No CUDA toolkit found" and fails with `'NoneType' object has no attribute
'CameraModelType'`. In your own terminal just `conda activate gsprune`. CPU tests need no GPU:
`~/miniforge3/envs/gsprune/bin/python -m pytest -q tests`. (Also saved in Claude's memory, `gpu-runs-easplatting`.)
Cluster (VACC) notes are archived in [`archive/vacc.md`](archive/vacc.md).

## 9. Run the EA experiments (GPU; from Claude Code wrap each in `gpu-run`, section 8)

Needs the scene data and NerfBaselines checkpoints (`bash scripts/get_data.sh mipnerf360 garden counter bicycle` and `... nb-checkpoints ...`, section 2). `--scene` is
a Mip-NeRF 360 scene name (bonsai and counter are indoor, `images_2`; the rest `images_4`). Importance scores are computed once per scene and mode and cached in
`results/cache/`. Defaults are in each tool's `--help`; these are the settings behind the tables in [`figures/README.md`](../figures/README.md):

```bash
python -m easplatting.experiment --scene bonsai                       # 4 genes, ga,random,sweep, 3 seeds, 8 views, mean PSNR >= 40 dB
python -m easplatting.experiment --scene bonsai --views 32 --worst-tau 35        # robust: 32 views, worst view >= 35 dB too
python -m easplatting.experiment --scene bonsai --views 32 --worst-tau 35 --importance max \
    --methods ga,random,cmaes,sweep --sweep-refine 12                 # main run (chosen score, CMA-ES, refined sweeps)
python -m easplatting.experiment --scene bonsai --views 32 --worst-tau 35 --importance density --methods sweep   # score screening
python -m easplatting.experiment --scene bonsai --views 32 --worst-tau 35 --importance max --fit-scale 1 \
    --methods ga,random,cmaes,sweep --sweep-refine 12 --seeds 10      # full-resolution fitness views (output folder gets _res1)
#   add --score-scale 1 to compute the importance scores at full res too (folder gets _sres1; matters on garden / bicycle)
python -m easplatting.rescore --runs bonsai_4g_v32_w35_imax          # re-score a finished run's genomes at full res, no new search
python -m easplatting.front --scene bonsai --views 32 --importance max           # NSGA-II / MAP-Elites / random / sweeps, size-quality fronts
```

Other options: `--genes 3` (thresholds only), `--view-select coverage` (fitness views by camera coverage), `--allowed-loss 0.5` (derive tau from an allowed real
loss vs the photos), `--parents/--children/--generations/--mutation` (GA budget: 32 + 64 x 10 = 672 evaluations), `--publish-images`. Importance modes: `sum`, `max`,
`density`, `lg`, `learned`, `fisher` (the last two did not work well, see `figures/README.md`).

**Outputs.** `results/<run>/` holds `log.jsonl` (one row per evaluation) and `summary.json` (best genome per run, held-out PSNRs) plus `renders.jpg` and the full-resolution
`images/`. The run name is `<scene>_<genes>g[_v<views>][_w<floor>][_i<mode>][_cov|_L<loss>]` (fronts: `<scene>_front_v<views>[_i<mode>]`). Each run also publishes
`plots.png` (or `front.png`), `renders.jpg` and `summary.json` into the tracked `figures/runs/<run>/` and refreshes `figures/compare.png`: **commit `figures/` with every run**.

Redo figures without rerunning a search: `python -m easplatting.plot_run --all` (plots + `compare.png`), `python -m easplatting.strip results/<run>` (renders; needs the GPU),
`python -m easplatting.plot_run results/<run> --images` (publish a run's full-resolution images, ~17 MB). After a run, update the tables in `figures/README.md`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| frustum check ~0% | gsplat-trained scene with raw COLMAP cameras: use the exported `cameras.json`; or wrong scene/cameras pair |
| PSNR < 15 dB, image looks shifted/garbage | wrong `--images` folder (intrinsics vs image size), or distorted (non-PINHOLE) COLMAP |
| PSNR a bit low, images look fine | background mismatch (NeRF-Synthetic: white), or `--rasterize-mode` vs how it was trained |
| GT size warning | `--images` doesn't match the resolution the scene was evaluated at |
| first render takes minutes | gsplat JIT compile (once); keep `TORCH_EXTENSIONS_DIR` set |
| "No CUDA GPUs are available" in a sandboxed shell | the sandbox hides `/dev/nvidia*`; use `gpu-run` (section 8) |
| LPIPS off vs a paper | different LPIPS convention/net; see section 7 (`--lpips-net vgg-raw` for Inria-style tables) |
| truck test PSNR >> 25 dB | Inria pretrained model saw the test views; use NB checkpoints for honest numbers |
