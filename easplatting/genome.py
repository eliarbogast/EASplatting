"""
genome.py — Gene representation, bounds, decoding, and mutation for EASplatting thresholds.
"""

import numpy as np
import torch
from easplatting import thresholds

class Genome:
    def __init__(self, values: np.ndarray, gene_bounds: list[tuple[float, float]]):
        """
        values: numpy array of raw gene values (e.g., [opacity_thresh, max_scale, anisotropy])
        gene_bounds: list of (min, max) tuples for each gene, including the outer 5% 'off' zones.
        """
        self.values = np.array(values, dtype=np.float32)
        self.gene_bounds = gene_bounds

    def decode_to_thresholds(self) -> dict:
        """
        Decodes raw gene values into keyword arguments for thresholds.keep_mask.
        The outer 5% of each gene's bound range triggers an explicit 'off' zone 
        (preventing accidental pruning of extreme Gaussians).
        """
        decoded = {}
        
        # Gene 0: Opacity threshold
        op_val, (op_min, op_max) = self.values[0], self.gene_bounds[0]
        off_threshold_op = op_max - 0.05 * (op_max - op_min)
        if op_val >= off_threshold_op:
            decoded['opacity_threshold'] = 0.0  # Turn pruning off
        else:
            decoded['opacity_threshold'] = float(op_val)

        # Gene 1: Max Scale threshold
        scale_val, (sc_min, sc_max) = self.values[1], self.gene_bounds[1]
        off_threshold_sc = sc_max - 0.05 * (sc_max - sc_min)
        if scale_val >= off_threshold_sc:
            decoded['max_scale'] = float('inf')  # Turn max scale pruning off
        else:
            decoded['max_scale'] = float(scale_val)

        # Gene 2: Anisotropy threshold
        aniso_val, (an_min, an_max) = self.values[2], self.gene_bounds[2]
        off_threshold_an = an_max - 0.05 * (an_max - an_min)
        if aniso_val >= off_threshold_an:
            decoded['anisotropy'] = float('inf')  # Turn anisotropy pruning off
        else:
            decoded['anisotropy'] = float(aniso_val)

        return decoded

    def to_mask(self, pcd_data) -> torch.Tensor:
        """Generates the keep mask over the scene using thresholds.keep_mask."""
        params = self.decode_to_thresholds()
        return thresholds.keep_mask(pcd_data, **params)

    def clone(self):
        return Genome(self.values.copy(), self.gene_bounds)

    @classmethod
    def random(cls, gene_bounds):
        """Initialize a random genome uniformly within the specified gene bounds."""
        values = np.array([np.random.uniform(b[0], b[1]) for b in gene_bounds], dtype=np.float32)
        return cls(values, gene_bounds)