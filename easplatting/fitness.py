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


def make_fitness_function(scene, stats, bounds, cameras, baselines, tau, worst_tau=None):
    """ scene: activated scene, stats: thresholds.gene_stats(scene), bounds: log10 range of each gene
        cameras, baselines: fitness views and their unpruned renders
        tau: mean PSNR (dB) the pruned renders must keep vs the baselines
        worst_tau: optional floor for the single worst view (a mean can hide one badly damaged view)
        returns fitness_function(genome) -> (fitness, kept fraction, mean PSNR)
    """
    def fitness_function(genome):
        mask = keep_mask(stats, *decode(genome, bounds))
        kept = mask.sum().item() / mask.numel()
        if mask.sum() < 1000: # degenerate, skip rendering
            return -999999.0, kept, 0.0
        rendered = render_views(prune(scene, mask), cameras)
        views = psnr_fast(rendered, baselines, max_db=60)
        psnr = views.mean().item()
        shortfall = max(tau - psnr, 0.0 if worst_tau is None else worst_tau - views.min().item())
        if shortfall <= 0: # feasible: fewer Gaussians is better
            return 1.0 - kept, kept, psnr
        return -1000.0 - shortfall, kept, psnr # infeasible: always worse than feasible
    return fitness_function


def make_objective_function(scene, stats, bounds, cameras, baselines):
    """ returns objective_function(genome) -> (kept, mean PSNR, worst PSNR, share)
        no constraint: the raw size and quality of a genome, for fronts and archives
        share: fraction of the removed Gaussians that the importance gene removes
    """

    def objective_function(genome):
        thresholds = decode(genome, bounds)
        mask = keep_mask(stats, *thresholds)
        kept = mask.sum().item() / mask.numel()
        if mask.sum() < 1000:  # degenerate, skip rendering
            return kept, 0.0, 0.0, 0.0
        rendered = render_views(prune(scene, mask), cameras)
        views = psnr_fast(rendered, baselines, max_db=60)

        share = 0.0
        if len(thresholds) > 3 and thresholds[3] > 0:
            removed = ~mask
            by_importance = (stats["importance"] < thresholds[3]) & removed
            share = by_importance.sum().item() / max(removed.sum().item(), 1)
        return kept, views.mean().item(), views.min().item(), share

    return objective_function
