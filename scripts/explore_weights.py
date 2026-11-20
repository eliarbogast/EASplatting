"""Which features do the evolved genomes weight, and do the weights agree across scenes and seeds?

    python scripts/explore_weights.py
Weights w = 2g - 1, normalized to unit length (the score is scale invariant); mean +- std over runs, and cosine similarity between runs.
"""

import json
from pathlib import Path

import numpy as np

W5 = "max,opacity,volume,views,camdist".split(",")
ALL = "max,opacity,volume,views,camdist,sum,density,lg,fisher,learned".split(",")
ALL2 = ALL + ["learned2", "fisherj"]
rows = [json.loads(l) for l in Path("results/explore/results.jsonl").read_text().splitlines()]
rows = [r for r in rows if r.get("learn_weights") and r["feasible"]]
for cfg in dict.fromkeys(r["name"] for r in rows):
    sel = [r for r in rows if r["name"] == cfg]
    names = sel[0]["features"].split(",")
    w = np.array([2 * np.array(r["genome"][: len(names)]) - 1 for r in sel])
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    scenes = [r["scene"] for r in sel]
    print(f"\n{cfg}  ({len(sel)} runs; scenes {sorted(set(scenes))})")
    print("  " + "".join(f"{n:>9}" for n in names))
    for scene in sorted(set(scenes)):
        m = w[[s == scene for s in scenes]]
        print(f"  {scene:<8}" if False else f"{scene:<9}" + "".join(f"{x:9.2f}" for x in m.mean(0)) + f"   (n={len(m)})")
    cos = w @ w.T
    iu = np.triu_indices(len(w), 1)
    same = [cos[i, j] for i, j in zip(*iu) if scenes[i] == scenes[j]]
    diff = [cos[i, j] for i, j in zip(*iu) if scenes[i] != scenes[j]]
    print(f"  cosine between runs: same scene {np.mean(same):.2f}" + (f", across scenes {np.mean(diff):.2f}" if diff else ""))
