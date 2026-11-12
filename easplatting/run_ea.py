"""run_ea.py — test run of the evolutionary search on bonsai:  python -m easplatting.run_ea"""

import numpy as np

from easplatting.cameras import load_cameras
from easplatting.ea import evolutionary_algorithm
from easplatting.fitness import make_fitness_function
from easplatting.genome import decode
from easplatting.gs_loader import activate, load_scene, prune
from easplatting.metrics import psnr_fast
from easplatting.render import render_views, select_indices
from easplatting.thresholds import gene_stats, keep_mask, slider_bounds


def main():
    scene_path = "pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply"
    cameras_path = "data/360_v2/bonsai"
    tau = 40.0 # PSNR (dB) vs the unpruned renders
    total_generations = 5
    num_parents = 16
    num_children = 32
    mutation_size = 0.1
    np.random.seed(0)

    print("Loading scene and cameras...")
    scene = activate(load_scene(scene_path, device="cuda"))
    stats = gene_stats(scene)
    b = slider_bounds(stats)
    bounds = [b["min_opacity"], b["max_scale"], b["max_anisotropy"]]
    cams = load_cameras(cameras_path, images="images_2")
    train_views = cams.subset(select_indices(len(cams), "train", k=8)) # 8 train views, evolved on half resolution
    fitness_views = train_views.scaled(0.5)
    baselines = render_views(scene, fitness_views)
    fitness_function = make_fitness_function(scene, stats, bounds, fitness_views, baselines, tau)

    print("Starting evolutionary search...")
    fitness, solutions, kept, psnr, diversity = evolutionary_algorithm(fitness_function=fitness_function,
        total_generations=total_generations, num_parents=num_parents, num_children=num_children,
        genome_length=3, mutation_size=mutation_size)
    for g in range(total_generations):
        print(f"gen {g}: fitness {fitness[g]:.4f}  kept {kept[g]:.3f}  PSNR {psnr[g]:.1f} dB  diversity {diversity[g]:.3f}")

    thresholds = decode(solutions[-1], bounds)
    print("best thresholds (min opacity, max scale, max anisotropy):", [float(f"{t:.4g}") for t in thresholds])
    full = render_views(prune(scene, keep_mask(stats, *thresholds)), train_views)
    print(f"same genome on the 8 views at full resolution: PSNR {psnr_fast(full, render_views(scene, train_views), max_db=60).mean().item():.1f} dB")


if __name__ == "__main__":
    main()
