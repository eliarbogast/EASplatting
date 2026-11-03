"""
check_env.py — will the pipeline work on THIS machine? One command, paste the output when asking for help.

    python scripts/check_env.py            # report only (safe anywhere, even before torch is installed)
    python scripts/check_env.py --smoke    # also compile gsplat's kernels and render one Gaussian (GPU needed)

Checks the OS, the NVIDIA driver and GPUs, whether the installed torch ships kernels for each GPU,
nvcc / host compiler (for gsplat's JIT build), required tools, Python packages, env vars and disk.
Prints OK / WARN / FAIL lines and a summary; exit code 1 if anything FAILed. Uses only the standard
library until it reaches the torch section, so it runs before setup_local.sh has been run.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GSPLAT_PIN = "1.5.3"

# ----------------------------------------------------------------------------- pure helpers


def parse_os_release(text):
    """PRETTY_NAME (or NAME VERSION) from the contents of /etc/os-release."""
    kv = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"')
    return kv.get("PRETTY_NAME") or " ".join(x for x in (kv.get("NAME"), kv.get("VERSION")) if x) or "unknown"


def parse_driver_cuda(banner):
    """'CUDA Version: 12.8' out of plain `nvidia-smi` output -> (12, 8), or None."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", banner)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_gpus(csv_text):
    """Rows of `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader`."""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, cc, mem, drv = parts[:4]
        m = re.match(r"(\d+)\.(\d+)$", cc)
        mem_mib = int(re.match(r"(\d+)", mem).group(1)) if re.match(r"\d+", mem) else None
        gpus.append(dict(name=name, cc=(int(m.group(1)), int(m.group(2))) if m else None,
                         mem_gb=mem_mib / 1024 if mem_mib else None, driver=drv))
    return gpus


def kernel_status(cc, arch_list):
    """
    Can a torch build with `arch_list` (torch.cuda.get_arch_list()) run on a GPU of compute
    capability `cc`? 'native' = SASS for this GPU (same major, minor <= GPU's), 'ptx' = only
    forward-compatible PTX (works, but slow first start), 'none' = "no kernel image" at runtime.
    """
    major, minor = cc
    for a in arch_list:
        m = re.match(r"sm_(\d+?)(\d)$", a)
        if m and int(m.group(1)) == major and int(m.group(2)) <= minor:
            return "native"
    for a in arch_list:
        m = re.match(r"compute_(\d+?)(\d)$", a)
        if m and (int(m.group(1)), int(m.group(2))) <= (major, minor):
            return "ptx"
    return "none"


# ------------------------------------------------------------------------------- reporting


class Report:
    def __init__(self):
        self.counts = {"OK": 0, "WARN": 0, "FAIL": 0}
        self.problems = []

    def section(self, title):
        print(f"\n== {title}")

    def add(self, level, msg, fix=None):
        self.counts[level] += 1
        tag = {"OK": "[ ok ]", "WARN": "[warn]", "FAIL": "[FAIL]", "INFO": "[    ]"}[level]
        print(f"{tag} {msg}")
        if level in ("WARN", "FAIL"):
            self.problems.append((level, msg))
            if fix:
                print(f"       -> {fix}")

    def info(self, msg):
        print(f"[    ] {msg}")


def dist_version(name):
    """Installed distribution version (for packages without __version__, e.g. plyfile)."""
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return "?"


def run(cmd, timeout=30, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


# --------------------------------------------------------------------------------- checks


def check_os(rep):
    rep.section("System")
    osr = Path("/etc/os-release")
    pretty = parse_os_release(osr.read_text()) if osr.exists() else platform.platform()
    libc = " ".join(platform.libc_ver()) or "unknown"
    rep.info(f"{pretty}, kernel {platform.release()}, {platform.machine()}, libc {libc}")
    rep.info(f"python {platform.python_version()} ({sys.executable})")
    if platform.system() != "Linux":
        rep.add("FAIL", f"{platform.system()} is not supported (real rendering needs Linux + NVIDIA)")
    elif platform.machine() != "x86_64":
        rep.add("WARN", f"{platform.machine()}: only x86_64 has been tested (conda CUDA packages / torch wheels differ)")
    else:
        rep.add("OK", "Linux x86_64")
    if sys.version_info < (3, 9):
        rep.add("FAIL", "Python < 3.9 (torch 2.8 needs 3.9+)", "use the conda env (setup_local.sh makes Python 3.10)")


def check_tools(rep):
    rep.section("Tools")
    # setup_local.sh needs git (trainer clone) plus coreutils; conda or a python3 venv for the env itself.
    for t, why, level in [("git", "cloning gsplat's trainer", "FAIL"), ("sed", "setup script", "FAIL"),
                          ("sort", "setup script", "FAIL"), ("conda", "creating the env + CUDA toolkit", "WARN")]:
        if shutil.which(t):
            rep.add("OK", f"{t}")
        else:
            fix = "install Miniforge (no root needed): https://github.com/conda-forge/miniforge" if t == "conda" else None
            rep.add(level, f"{t} not found ({why})", fix)
    rep.info("wget / curl / unzip are NOT needed (scripts/download.py handles downloads)")


def check_gpu(rep):
    rep.section("NVIDIA driver and GPUs")
    if not shutil.which("nvidia-smi"):
        rep.add("FAIL", "nvidia-smi not found: no NVIDIA driver on this host",
                "on a cluster login node this is expected: build the env there with NO_GPU_OK=1 and run "
                "this script again on a GPU node")
        return []
    rc, banner = run(["nvidia-smi"])
    drv_cuda = parse_driver_cuda(banner)
    rc2, csv = run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"])
    if rc2 != 0:   # older drivers lack compute_cap
        rc3, csv = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        csv = "\n".join(l.replace(",", ", n/a,", 1) for l in csv.splitlines()) if rc3 == 0 else ""
    gpus = parse_gpus(csv)
    if rc != 0 or not gpus:
        rep.add("FAIL", "nvidia-smi is present but reports no GPU (driver not loaded? no GPU allocated?)",
                "on SLURM request one: --gres=gpu:1 or --constraint (see docs/CLUSTER.md)")
        return []
    for i, g in enumerate(gpus):
        cc = f"sm_{g['cc'][0]}{g['cc'][1]}" if g["cc"] else "sm_?"
        rep.info(f"GPU {i}: {g['name']}, {cc}, {g['mem_gb']:.0f} GB, driver {g['driver']}")
        if g["cc"] and g["cc"] < (7, 0):
            rep.add("WARN", f"GPU {i} is older than Volta ({cc}): torch 2.8/cu128 has no kernels for it",
                    "setup_local.sh switches to torch 2.4.1/cu124 automatically for this case")
        if g["mem_gb"] and g["mem_gb"] < 8:
            rep.add("WARN", f"GPU {i} has {g['mem_gb']:.0f} GB: indoor scenes need ~5 GB at batch 4, "
                            "outdoor scenes are untested; lower render batch_size if you run out")
    if drv_cuda is None:
        rep.add("WARN", "could not read the driver's CUDA version from nvidia-smi")
    elif drv_cuda < (12, 4):
        rep.add("FAIL", f"driver supports CUDA {drv_cuda[0]}.{drv_cuda[1]} only; torch cu124/cu128 need >= 12.4",
                "update the NVIDIA driver, or override: CU=cu118 TORCH=2.4.1 TORCHVISION=0.19.1 bash scripts/setup_local.sh (untested)")
    elif drv_cuda < (12, 8):
        rep.add("OK", f"driver CUDA {drv_cuda[0]}.{drv_cuda[1]}: setup will use torch 2.4.1/cu124 (cu128 needs 12.8+; "
                      "Blackwell GPUs need a newer driver)")
    else:
        rep.add("OK", f"driver CUDA {drv_cuda[0]}.{drv_cuda[1]} (torch 2.8/cu128 supported)")
    return gpus


def check_setup_choice(rep):
    rep.section("What setup_local.sh would install here")
    env = dict(os.environ, PRINT_CONFIG="1", NO_GPU_OK="1")
    rc, out = run(["bash", str(ROOT / "scripts" / "setup_local.sh")], env=env)
    cfg = [l for l in out.splitlines() if l.startswith(("config:", "!!"))]
    rep.info(cfg[-1] if cfg else f"(could not determine: exit {rc})")


def check_compilers(rep):
    rep.section("CUDA toolkit and compiler (gsplat JIT-compiles its kernels on first use)")
    nvcc = shutil.which("nvcc") or (str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc") if os.environ.get("CUDA_HOME") else None)
    if nvcc and Path(nvcc).exists():
        _, out = run([nvcc, "--version"])
        m = re.search(r"release (\d+\.\d+)", out)
        rep.add("OK", f"nvcc {m.group(1) if m else '?'} ({nvcc})")
    else:
        rep.add("WARN", "nvcc not found (fine before setup; setup_local.sh installs it into the conda env)",
                "or `module load cuda` on a cluster")
    cc = os.environ.get("CC") or shutil.which("x86_64-conda-linux-gnu-cc") or shutil.which("gcc") or shutil.which("cc")
    if cc:
        rep.add("OK", f"host C compiler: {cc}")
    else:
        rep.add("WARN", "no host C compiler found (the conda env provides one after setup)")
    ninja = shutil.which("ninja")
    rep.add("OK" if ninja else "WARN", "ninja" if ninja else "ninja not found (pip install ninja; setup does this)")


def check_python_env(rep, gpus, smoke):
    rep.section("Python packages")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", f"torch is not importable ({type(e).__name__}: {e})",
                "run: bash scripts/setup_local.sh   (then activate the env)")
        return
    rep.info(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    cuda_ok = torch.cuda.is_available()
    arch = torch.cuda.get_arch_list() if torch.version.cuda else []
    rep.info(f"torch kernels for: {' '.join(arch) or 'none (CPU-only build)'}")
    if not torch.version.cuda:
        rep.add("FAIL", "this torch is a CPU-only build", "reinstall with the CUDA index (setup_local.sh does)")
    elif not cuda_ok:
        rep.add("FAIL", "torch.cuda.is_available() is False", "driver/GPU problem, or no GPU allocated on this node")
    else:
        for i in range(torch.cuda.device_count()):
            cc = torch.cuda.get_device_capability(i)
            st = kernel_status(cc, arch)
            label = f"GPU {i} ({torch.cuda.get_device_name(i)}, sm_{cc[0]}{cc[1]})"
            if st == "native":
                rep.add("OK", f"{label}: torch has native kernels")
            elif st == "ptx":
                rep.add("WARN", f"{label}: only PTX fallback (works, slow first start)")
            else:
                rep.add("FAIL", f"{label}: torch {torch.__version__} has no kernels for this GPU",
                        "Pascal or older -> torch 2.4.1/cu124 (setup_local.sh picks it); brand-new GPU -> newer torch/cu128+")
    for mod, want, level in [("gsplat", GSPLAT_PIN, "FAIL"), ("torchmetrics", None, "FAIL"), ("plyfile", None, "FAIL"),
                             ("PIL", None, "FAIL"), ("numpy", None, "FAIL"), ("viser", None, "WARN"), ("nerfview", None, "WARN")]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None) or dist_version({"PIL": "pillow"}.get(mod, mod))
            if want and v != want:
                rep.add("WARN", f"{mod} {v} (project pins {want})")
            else:
                rep.add("OK", f"{mod} {v}")
            if mod == "numpy" and int(str(v).split(".")[0]) >= 2:
                rep.add("WARN", "numpy >= 2 (project pins numpy<2)")
        except Exception as e:  # noqa: BLE001
            extra = " (only needed for viewer.py)" if mod in ("viser", "nerfview") else ""
            rep.add(level, f"{mod} not importable{extra}: {type(e).__name__}", "bash scripts/setup_local.sh")
    if smoke:
        if not cuda_ok:
            rep.add("FAIL", "--smoke needs a working GPU")
        else:
            rep.info("running scripts/smoke_gsplat.py (compiles gsplat on first use: ~1-5 minutes) ...")
            rc, out = run([sys.executable, str(ROOT / "scripts" / "smoke_gsplat.py")], timeout=3600)
            last = (out.strip().splitlines() or ["(no output)"])[-1]
            rep.add("OK" if rc == 0 else "FAIL", f"gsplat render: {last}")
    else:
        rep.info("(run with --smoke to also compile gsplat and render a test image)")


def check_env_vars_and_disk(rep):
    rep.section("Environment variables and disk")
    for k in ("TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR", "TORCH_HOME", "CUDA_HOME", "CUDA_VISIBLE_DEVICES"):
        rep.info(f"{k} = {os.environ.get(k, '(unset)')}")
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        rep.add("WARN", "TORCH_EXTENSIONS_DIR is unset: kernels build under ~/.cache/torch_extensions",
                "export TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions (on a cluster: a shared directory)")
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < 15:
        rep.add("WARN", f"only {free:.0f} GB free here; the default data set is ~4 GB, each scene checkpoint ~1.4 GB")
    else:
        rep.add("OK", f"{free:.0f} GB free in {ROOT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="also compile gsplat and render one Gaussian (needs a GPU)")
    args = ap.parse_args()

    rep = Report()
    print("EASplatting environment check")
    check_os(rep)
    check_tools(rep)
    gpus = check_gpu(rep)
    check_setup_choice(rep)
    check_compilers(rep)
    check_python_env(rep, gpus, args.smoke)
    check_env_vars_and_disk(rep)

    c = rep.counts
    print(f"\n== Summary: {c['OK']} ok, {c['WARN']} warnings, {c['FAIL']} failures")
    for level, msg in rep.problems:
        print(f"   {level}: {msg}")
    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()
