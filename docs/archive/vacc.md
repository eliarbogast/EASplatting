# VACC notes (archived 2026-11-17)

We are probably not going to use the UVM VACC, so its material was moved out of the working docs. Nothing here is
verified against the cluster (written from its public docs before we had access). `scripts/setup_local.sh`
(`NO_GPU_OK=1`), `check_env.py --smoke-only` / `--prefetch-weights` and `get_data.sh` still carry the cluster-oriented
options described below.

## From RUNNING.md (was section 8)

### Running on the UVM VACC

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

## From BACKGROUND.md (was section 9)

### Compute and the VACC

#### 9.1 Don't use a SLURM task per individual
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

#### 9.2 Practicalities
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

## From EA_PLAN.md and BACKGROUND.md (planned, never built)
- VACC job scripts: one process per (scene x method), seeds sequentially, a SLURM array over those, a persistent
  `TORCH_EXTENSIONS_DIR`; `slurm/array.sbatch` over configs x seeds.
- Re-run `scripts/measure.py cost` on each VACC GPU type; one GPU SKU for every timed comparison.
- TensorBoard / viewer over an ssh port-forward (`ssh -L 8080:<gpu-node>:8080 <vacc-login>`).
