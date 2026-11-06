# Running on the UVM VACC (checklist)

Written before we have access, from the VACC docs ([cluster specs](https://www.uvm.edu/vacc/cluster-specs),
[partitions](https://www.uvm.edu/vacc/docs/beyond_basics/slurm_partitions/),
[submitting jobs](https://www.uvm.edu/vacc/docs/start_guide/running_jobs/)). Verify against `sinfo` /
`show_node_constraints` once we're on; the "Confirm with VACC" list at the end is what the docs don't say.

## What the VACC has

| GPU | Arch (`TORCH_CUDA_ARCH_LIST`) | Memory | Notes |
|---|---|---|---|
| H200 (NVL, SXM) | 9.0 | 141 GB | most GPUs; 4-8 per node |
| H100 | 9.0 | 80 GB | 2 nodes |
| A100 | 8.0 | 40 GB | 2 nodes; this is what `gpu-debug` gives |
| RTX Pro 6000 (Blackwell) | 12.0 | 96 GB | same architecture as Eli's desktop |

RHEL 9.6, Slurm, Lmod (CUDA 11/12/13 modules), shared GPFS storage. Partitions: `nvgpu` (GPU jobs,
48 h), `gpu-debug` (3 h, 1 A100, max 2 submitted / 1 running per user), `gpu-preempt` (H200 nodes,
**jobs can be cancelled**). `general` has no GPUs. Pick GPUs with `--constraint`, e.g.
`GPU_SKU:H200`, `GPU_MEM:80GB`, `gpu_any`.

Good news: the torch 2.8 + cu128 wheels carry sm_80, sm_90 and sm_120 kernels, so **one env and one
gsplat build serve every node type**.

## Checklist

0. **Look before you build:** `python scripts/check_env.py` on the login node and again on a GPU node
   (`--smoke` once the env exists); it reports the OS, driver, GPUs, which torch build would be used,
   compilers, missing tools and env vars. Paste its output if anything looks off.
1. **Build the env on a login node** (no GPU needed), on project/GPFS storage, not `$HOME` (the conda
   CUDA toolkit alone is several GB). No root needed; Miniforge installs into a directory you own.
   ```bash
   NO_GPU_OK=1 TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" bash scripts/setup_local.sh
   ```
2. **Pre-warm gsplat's JIT cache once** on a GPU node (a `gpu-debug` job is enough). Use the same arch
   list and a shared cache in every job, otherwise a different list triggers a rebuild and, in an
   array, tasks race on the same build lock:
   ```bash
   export TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" TORCH_EXTENSIONS_DIR=/path/on/gpfs/torch_ext
   python scripts/smoke_gsplat.py      # expect ~[0.9, 0.45, 0.18]; compile takes about a minute
   ```
3. **Make compute nodes self-sufficient** (they may be offline): fetch data and the LPIPS weights on a
   login node, then export the same `TORCH_HOME` in jobs.
   ```bash
   bash scripts/get_data.sh mipnerf360 bonsai counter ...   # + nb-checkpoints, see LOCAL_PIPELINE.md
   TORCH_HOME=/path/on/gpfs/torch_home python scripts/prefetch_weights.py
   ```
4. **Job preamble** (same in every job): activate the env, export `TORCH_EXTENSIONS_DIR`,
   `TORCH_CUDA_ARCH_LIST`, `TORCH_HOME`; request one GPU with a `--constraint`. Give each array task its
   own log/JSONL file (concurrent appends corrupt a shared file). Check that the node's driver
   supports CUDA 12.8 (`nvidia-smi`, driver >= 570); the CUDA 13 modules suggest it does.
5. **One GPU SKU for every timed comparison**, e.g. `--constraint=GPU_SKU:H200`. Re-measure
   `timing.render_s_per_view` there (the budget in `EA_PLAN.md` is from a 5070 Ti) and compare methods by
   evaluation count, not wall time. `gpu-debug` (A100) is for dry runs, not timings.
6. **Runs must be resumable** (skip finished seeds, write per-seed results as you go): required for
   `gpu-preempt`, and good practice for `nvgpu`. Our runs are minutes long, so seed-level resume is enough.
7. **Parallelism = number of GPUs.** On the 5070 Ti, 1 / 2 / 4 processes sharing a GPU gave 76 / 71 / 74
   evals/s in total, so one process already saturates it (`scripts/bench_eval.py`, part B). Re-run that
   benchmark on an H200 before assuming a bigger GPU behaves differently; until then spread seeds across
   GPUs (array tasks), not across processes on one GPU. Memory is ~1.1 GB per million Gaussians at 8 views.
8. **Logs:** progress meters should print plain periodic lines when not attached to a TTY; plotting must
   use matplotlib's `Agg` backend.
9. **Viewer on a node:** `python viewer.py ...` binds to 127.0.0.1 only. From a laptop:
   `ssh -L 8080:<gpu-node>:8080 <vacc-login>`, then open `http://localhost:8080`.

## Confirm with VACC (not stated in the docs we found)

- Do compute nodes have internet? (Step 3 assumes not.)
- Exact GPU request syntax: the docs' example uses only `--constraint`; is `--gres=gpu:1` also needed?
- Scratch/project storage paths, quotas and purge policy for the data (~2 GB per scene, ~1.4 GB per checkpoint).
- Which conda / Python / compiler modules exist (we assume none and install Miniforge ourselves).
- Per-user limits on jobs and array tasks for `nvgpu` (only `gpu-debug` limits are published).
