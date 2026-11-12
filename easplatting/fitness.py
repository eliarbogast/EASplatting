"""fitness.py — feasibility-first fitness for pruning genomes."""

import numpy as np

from easplatting.genome import decode
from easplatting.gs_loader import prune
from easplatting.metrics import psnr_fast
from easplatting.render import render_views
from easplatting.thresholds import keep_mask


class Individual:

    def __init__(self, fitness_function, genome_length):
        self.genome = np.random.rand(genome_length)
        self.fitness_function = fitness_function
        self.fitness = 0
        self.kept = 1.0
        self.psnr = 0.0

    def eval_fitness(self):
        self.fitness, self.kept, self.psnr = self.fitness_function(self.genome)


def make_fitness_function(scene, stats, bounds, cameras, baselines, tau):
    """ scene: activated scene, stats: thresholds.gene_stats(scene), bounds: log10 range of each gene
        cameras, baselines: fitness views and their unpruned renders
        tau: PSNR (dB) the pruned renders must keep vs the baselines
    """
    def fitness_function(genome):
        mask = keep_mask(stats, *decode(genome, bounds))
        kept = mask.sum().item() / mask.numel()
        if mask.sum() < 1000: # degenerate, skip rendering
            return -999999.0, kept, 0.0
        rendered = render_views(prune(scene, mask), cameras)
        psnr = psnr_fast(rendered, baselines, max_db=60).mean().item()
        if psnr >= tau: # feasible: fewer Gaussians is better
            return 1.0 - kept, kept, psnr
        return -1000.0 - (tau - psnr), kept, psnr # infeasible: always worse than feasible
    return fitness_function
