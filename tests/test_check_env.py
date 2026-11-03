"""scripts/check_env.py: parsers, the kernel-compatibility rule, and the script end to end (stubbed nvidia-smi)."""

import os
import subprocess
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from check_env import kernel_status, parse_driver_cuda, parse_gpus, parse_os_release  # noqa: E402

CU128 = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
CU124 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def test_kernel_status_rule():
    assert kernel_status((12, 0), CU128) == "native"
    assert kernel_status((8, 9), CU128) == "native"        # Ada runs the sm_86 cubin (same major, lower minor)
    assert kernel_status((9, 0), CU124) == "native"
    assert kernel_status((12, 0), CU124) == "none"         # Blackwell on an old torch: "no kernel image"
    assert kernel_status((6, 1), CU128) == "none"          # Pascal on torch 2.8/cu128
    assert kernel_status((6, 1), CU124) == "native"        # Pascal is fine on the cu124 build (sm_60 cubin)
    assert kernel_status((12, 0), CU124 + ["compute_90"]) == "ptx"
    assert kernel_status((7, 5), []) == "none"


def test_parsers():
    assert parse_driver_cuda("| NVIDIA-SMI 595.1  Driver Version: 595.1  CUDA Version: 13.2 |") == (13, 2)
    assert parse_driver_cuda("garbage") is None
    g = parse_gpus("NVIDIA GeForce RTX 5070 Ti, 12.0, 16303 MiB, 595.104.02\nTesla V100-SXM2, 7.0, 32768 MiB, 535.1\n")
    assert g[0]["cc"] == (12, 0) and g[0]["driver"] == "595.104.02" and round(g[0]["mem_gb"]) == 16
    assert g[1]["name"] == "Tesla V100-SXM2" and g[1]["cc"] == (7, 0)
    assert parse_gpus("A, n/a, 1024 MiB, 400.0")[0]["cc"] is None
    assert parse_os_release('NAME="Red Hat Enterprise Linux"\nVERSION="9.6"\nPRETTY_NAME="Red Hat Enterprise Linux 9.6 (Plow)"\n').startswith("Red Hat")
    assert parse_os_release('NAME="X"\nVERSION="1"\n') == "X 1"


def _stub(tmp_path, cc, cuda, exit_code=0):
    """A fake nvidia-smi that, like the real one, prints only the --query-gpu columns asked for."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    f = d / "nvidia-smi"
    f.write_text(f"""#!/bin/sh
[ {exit_code} -ne 0 ] && exit {exit_code}
case "$*" in
  *--query-gpu=*)
    q=${{*#*--query-gpu=}}; q=${{q%% *}}; out=""
    for field in $(echo "$q" | tr , ' '); do
      case $field in
        name) v="NVIDIA Stub GPU";; compute_cap) v="{cc}";; memory.total) v="16384 MiB";;
        driver_version) v="999.9";; *) v="?";;
      esac
      out="$out${{out:+, }}$v"
    done
    echo "$out";;
  *) echo "| NVIDIA-SMI 999.9   Driver Version: 999.9   CUDA Version: {cuda} |";;
esac
""")
    f.chmod(0o755)
    return str(d)


def _run(tmp_path, *stub_args, **kw):
    env = dict(os.environ, PATH=_stub(tmp_path, *stub_args, **kw) + os.pathsep + os.environ["PATH"])
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_env.py")], capture_output=True, text=True, env=env)


def test_script_runs_and_reports_gpu(tmp_path):
    r = _run(tmp_path, "9.0", "12.8")
    assert "Traceback" not in r.stderr and "== Summary" in r.stdout
    assert "sm_90" in r.stdout and "driver CUDA 12.8" in r.stdout
    assert "torch 2.8.0 (cu128)" in r.stdout          # what setup_local.sh would choose for an H100


def test_script_flags_old_driver_and_pascal(tmp_path):
    old = _run(tmp_path, "7.5", "11.8")
    assert old.returncode == 1 and "need >= 12.4" in old.stdout
    pascal = _run(tmp_path, "6.1", "12.8")
    assert "older than Volta" in pascal.stdout and "torch 2.4.1 (cu124)" in pascal.stdout


def test_script_fails_cleanly_without_gpu(tmp_path):
    r = _run(tmp_path, "9.0", "12.8", exit_code=9)       # nvidia-smi present but failing
    assert r.returncode == 1 and "Traceback" not in r.stderr and "no GPU" in r.stdout
