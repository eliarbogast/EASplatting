"""genome.py — turn a genome (3 numbers in [0, 1]) into pruning thresholds."""

import numpy as np


def decode(genome, bounds, dead_zone=0.05):
    """ genome: opacity, max scale, max anisotropy as fractions of their log10 ranges (bounds)
        the outer 5% of a range means the gene is off (opacity: bottom, scale and anisotropy: top)
        returns: min_opacity, max_scale, max_anisotropy for thresholds.keep_mask
    """
    min_opacity, max_scale, max_anisotropy = [10 ** (low + g * (high - low)) for g, (low, high) in zip(genome, bounds)]
    if genome[0] <= dead_zone: min_opacity = 0.0
    if genome[1] >= 1 - dead_zone: max_scale = np.inf
    if genome[2] >= 1 - dead_zone: max_anisotropy = np.inf
    return min_opacity, max_scale, max_anisotropy
