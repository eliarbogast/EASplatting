"""Plot results/explore/fronts.jsonl: quality vs % kept on the TEST views, scores as curves and evolved genomes as markers.

    python scripts/plot_explore_fronts.py        # writes figures/explore_fronts.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in Path("results/explore/fronts.jsonl").read_text().splitlines()]
scenes = list(dict.fromkeys(r["scene"] for r in rows))
fig, axes = plt.subplots(2, len(scenes), figsize=(6.2 * len(scenes), 7.5), squeeze=False)
colors = dict(max="#1b6ca8", sum="#2a9d8f", lg="#d1495b", density="#edae49", opacity="#8d8d8d", fisherj="#7b4fa0", learned2="#c06c2b")
for j, scene in enumerate(scenes):
    R = [r for r in rows if r["scene"] == scene]
    un = next(r for r in R if r["kind"] == "unpruned")
    for label in dict.fromkeys(r["label"] for r in R if r["kind"] == "score"):
        pts = sorted((r for r in R if r["kind"] == "score" and r["label"] == label), key=lambda r: r["kept"])
        x = [100 * r["kept"] for r in pts]
        axes[0][j].plot(x, [r["photo_psnr"] - un["photo_psnr"] for r in pts], "-o", ms=3, color=colors.get(label), label=label)
        axes[1][j].plot(x, [r["test_worst"] for r in pts], "-o", ms=3, color=colors.get(label), label=label)
    ev = [r for r in R if r["kind"] == "evolved"]
    axes[0][j].scatter([100 * r["kept"] for r in ev], [r["photo_psnr"] - un["photo_psnr"] for r in ev], marker="*", s=70, color="black", zorder=5, label="evolved genomes")
    axes[1][j].scatter([100 * r["kept"] for r in ev], [r["test_worst"] for r in ev], marker="*", s=70, color="black", zorder=5, label="evolved genomes")
    axes[0][j].set_title(f"{scene}: test PSNR vs photos, change from unpruned ({un['photo_psnr']:.1f} dB)")
    axes[1][j].set_title(f"{scene}: worst test view PSNR vs the unpruned render")
    axes[0][j].set_ylabel("dB vs unpruned")
    axes[1][j].set_ylabel("dB")
    axes[1][j].axhline(35, color="k", lw=0.6, ls="--")
    for i in (0, 1):
        axes[i][j].set_xlabel("% of Gaussians kept")
        axes[i][j].grid(alpha=0.25)
    axes[0][j].set_ylim(-5, 0.3)
    axes[1][j].set_ylim(10, 50)
axes[0][0].legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("figures/explore_fronts.png", dpi=130)
print("wrote figures/explore_fronts.png")
