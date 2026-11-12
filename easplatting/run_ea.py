import torch
import numpy as np
from pathlib import Path
from easplatting.gs_loader import load_scene
from easplatting.cameras import load_cameras, select_indices
from easplatting.fitness import Individual
# Import your evolutionary_algorithm function here

def main():
    device = "cuda"
    scene_path = "pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply"
    cameras_path = "data/360_v2/bonsai"
    baseline_pt = "results/baseline_bonsai/baselines_train.pt"
    
    print("Loading scene and cameras...")
    scene_data = load_scene(scene_path, device=device)
    cams = load_cameras(cameras_path, images="images_2", split="test") # or train split
    train_idx = select_indices(len(cams), "train", every=1)
    train_cameras = cams.subset(train_idx)
    
    print("Loading cached unpruned baselines...")
    ckpt = torch.load(baseline_pt, map_location=device, weights_only=False)
    baseline_renders = ckpt["images"]
    
    # Define fitness function wrapper for Individual
    fitness_cache = {}
    tau = 40.0  # PSNR constraint threshold vs unpruned baseline
    
    def fitness_wrapper(genome, scene, baselines, cams_ref, tau_val):
        ind = Individual(genome_length=3)
        ind.genome = genome
        # Calls the evaluation logic we set up in fitness.py
        return ind.eval_fitness(scene, baselines, cams_ref, tau_val, fitness_cache)

    print("Starting Evolutionary Search smoke test...")
    # Run a short smoke test (e.g., 5 generations, small pop)
    fitness_history, solutions_history = evolutionary_algorithm(
        fitness_function=fitness_wrapper,
        total_generations=5,
        num_parents=16,
        num_children=32,
        genome_length=3,
        mutation_size=0.1,
        crossover=True
    )
    
    print("EA run complete! Best fitness history:", fitness_history)

if __name__ == "__main__":
    main()