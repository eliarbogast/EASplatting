"""Summarize results/explore/results.jsonl: mean % kept per scene / config / method (lower is better), with held-out numbers.

    python scripts/summarize_explore.py
"""

import collections
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path("results/explore/results.jsonl").read_text().splitlines()]
rows = [r for r in rows if r["method"] != "sweep"]
sweeps = {r["scene"]: r["sweep_kept"] for r in rows if r.get("sweep_kept")}
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["scene"], r["name"], r["method"])].append(r)

order = []
for r in rows:
    if r["name"] not in order:
        order.append(r["name"])
for scene in sorted({r["scene"] for r in rows}):
    print(f"\n{scene}: 1-D importance sweep {100 * sweeps[scene]:.1f}% kept")
    print(f"{'config':<16}{'method':<8}{'n':>2} {'kept %':>7} {'best':>6} {'test dB':>8} {'test worst':>11} {'train worst':>12} {'evals>sweep':>12}")
    for name in order:
        for method in ("ga", "cmaes", "random"):
            g = groups.get((scene, name, method))
            if not g:
                continue
            ok = [r for r in g if r["feasible"]]
            if not ok:
                print(f"{name:<16}{method:<8}{len(g):>2}   infeasible")
                continue
            mean = lambda k: sum(r[k] for r in ok) / len(ok)
            first = [r["evals_to_beat_sweep"] for r in ok if r.get("evals_to_beat_sweep") is not None]
            print(f"{name:<16}{method:<8}{len(ok):>2} {100 * mean('kept'):7.1f} {100 * min(r['kept'] for r in ok):6.1f} "
                  f"{mean('test_psnr'):8.2f} {mean('test_worst'):11.2f} {mean('train_worst'):12.2f} "
                  f"{(sum(first) / len(first) if first else float('nan')):12.0f}")
