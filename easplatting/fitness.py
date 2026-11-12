"""
fitness.py — Feasibility-first evaluation, rendering wrapper, and caching for EASplatting.
"""

import numpy as np
import torch
from easplatting.genome import Genome
from easplatting.render import render_views
from easplatting.metrics import psnr_fast

class Individual:
    def __init__(self, fitness_function=None, genome_length=3, gene_bounds=None):
        """
        Represents an individual in the evolutionary population.
        """
        # Default typical ranges for opacity, max scale, and anisotropy
        self.gene_bounds = gene_bounds or [(0.0, 0.05), (0.0, 10.0), (1.0, 100.0)]
        self.genome = Genome.random(self.gene_bounds)
        self.fitness_function = fitness_function
        
        self.fitness = -999999.0
        self.kept_fraction = 1.0
        self.psnr = 0.0
        self.feasible = False
        self.evaluated = False

    def eval_fitness(self, scene_data, baseline_renders, train_cameras, tau, fitness_cache=None):
        """
        Evaluates the individual's fitness using the provided scene, baseline renders, 
        training cameras, and PSNR threshold constraint (tau).
        """
        if self.evaluated:
            return self.fitness

        # Key for caching based on rounded genome values to avoid redundant renders
        cache_key = tuple(np.round(self.genome.values, decimals=4))
        if fitness_cache is not None and cache_key in fitness_cache:
            self.fitness, self.kept_fraction, self.psnr, self.feasible = fitness_cache[cache_key]
            self.evaluated = True
            return self.fitness

        # Execute evaluation via fitness function if supplied, or default logic
        if self.fitness_function is not None:
            self.fitness, self.kept_fraction, self.psnr, self.feasible = self.fitness_function(
                self.genome, scene_data, baseline_renders, train_cameras, tau
            )
        else:
            # Default fallback evaluation logic
            mask = self.genome.to_mask(scene_data)
            kept_count = int(mask.sum().item())
            total_count = scene_data["means"].shape[0]
            self.kept_fraction = kept_count / total_count

            # Short-circuit degenerate/empty masks
            if kept_count < 1000:
                self.psnr = 0.0
                self.feasible = False
                self.fitness = -999999.0
            else:
                # Prune raw scene dictionary
                pruned_scene = {k: (v[mask] if torch.is_tensor(v) and v.ndim > 0 else v) 
                                for k, v in scene_data.items()}
                
                # Render fitness views at half resolution (e.g., train views)
                from easplatting.gs_loader import activate
                activated_pruned = activate(pruned_scene)
                
                rendered = render_views(
                    activated_pruned, train_cameras, list(range(len(train_cameras))), 
                    rasterize_mode="classic"
                )
                
                # Compute PSNR against cached unpruned baseline renders (capped at 60 dB)
                psnr_vals = psnr_fast(rendered, baseline_renders.to(rendered.device).float(), max_db=60)
                self.psnr = float(psnr_vals.mean().item())

                # Feasibility-First Assessment
                if self.psnr >= tau:
                    # Feasible: maximize compression (minimize kept fraction)
                    self.fitness = 1.0 - self.kept_fraction
                    self.feasible = True
                else:
                    # Infeasible: heavy penalty proportional to PSNR deficit
                    self.fitness = -1000.0 - (tau - self.psnr)
                    self.feasible = False

        if fitness_cache is not None:
            fitness_cache[cache_key] = (self.fitness, self.kept_fraction, self.psnr, self.feasible)

        self.evaluated = True
        return self.fitness