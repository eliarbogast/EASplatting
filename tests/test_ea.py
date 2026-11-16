"""genome decoding, the fitness function (fake rasterizer) and the evolutionary algorithm, on CPU."""

import numpy as np
import torch

from conftest import H, W
from easplatting.cameras import Cameras
from easplatting.ea import evolutionary_algorithm
from easplatting.fitness import Individual, make_fitness_function
from easplatting.genome import decode
from easplatting.gs_loader import activate
from easplatting.render import render_views
from easplatting.thresholds import gene_stats, slider_bounds

BOUNDS = [(-3.0, -0.3), (-1.5, 0.5), (0.0, 4.0)]


def test_decode_off_zones_and_log_ranges():
    opacity, scale, aniso = decode([0.5, 0.5, 0.5], BOUNDS)
    assert np.isclose(opacity, 10 ** -1.65) and np.isclose(scale, 10 ** -0.5) and np.isclose(aniso, 10 ** 2.0)
    assert decode([0.0, 0.5, 0.5], BOUNDS)[0] == 0.0 and decode([1.0, 0.5, 0.5], BOUNDS)[0] > 0.4   # opacity: off at the bottom
    assert decode([0.5, 1.0, 1.0], BOUNDS)[1:] == (np.inf, np.inf)                                    # scale, aniso: off at the top
    assert np.isfinite(decode([0.5, 0.0, 0.0], BOUNDS)).all()


def make_scene_fitness(raw_scene, tau):
    scene = activate(raw_scene)
    stats = gene_stats(scene)
    b = slider_bounds(stats)
    bounds = [b["min_opacity"], b["max_scale"], b["max_anisotropy"]]
    viewmat = torch.eye(4)
    viewmat[2, 3] = 3.0
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    cams = Cameras(names=["v"], viewmats=viewmat[None], Ks=K[None], widths=[W], heights=[H])
    return make_fitness_function(scene, stats, bounds, cams, render_views(scene, cams), tau)


def test_fitness_is_feasibility_first(raw_scene):
    all_off = np.array([0.0, 1.0, 1.0])                                  # nothing pruned: identical render
    fitness, kept, psnr = make_scene_fitness(raw_scene, tau=40.0)(all_off)
    assert kept == 1.0 and psnr == 60 and fitness == 0.0                  # feasible: 1 - kept
    fitness, kept, psnr = make_scene_fitness(raw_scene, tau=100.0)(all_off)
    assert fitness == -1000.0 - 40.0                                      # infeasible: always below any feasible one
    fitness, kept, psnr = make_scene_fitness(raw_scene, tau=40.0)(np.array([1.0, 0.0, 0.0]))
    assert fitness == -999999.0 and kept < 0.25                           # near-empty mask is skipped


def test_evolutionary_algorithm_improves_toy_fitness():
    target = np.array([0.7, 0.2, 0.9])
    toy = lambda genome: (-np.sum(np.square(genome - target)), 0.5, 40.0)
    np.random.seed(0)
    fitness, solutions, kept, psnr, diversity = evolutionary_algorithm(fitness_function=toy, total_generations=30,
        num_parents=10, num_children=20, genome_length=3, mutation_size=0.1)
    assert fitness.shape == (30,) and solutions.shape == (30, 3) and kept.shape == psnr.shape == diversity.shape == (30,)
    assert np.all(np.diff(fitness) >= 0) and fitness[-1] > fitness[0]     # elite is kept: never gets worse
    assert fitness[-1] > -0.01 and ((solutions >= 0) & (solutions <= 1)).all()


def test_individual_stores_fitness_function_outputs():
    ind = Individual(lambda genome: (1.5, 0.3, 41.0), 3)
    ind.eval_fitness()
    assert (ind.fitness, ind.kept, ind.psnr) == (1.5, 0.3, 41.0) and ind.genome.shape == (3,)


def test_cameras_scaled(colmap_scene):
    from easplatting.cameras import load_colmap
    cams = load_colmap(colmap_scene[0], images="images_2")
    half = cams.scaled(0.5)
    assert half.size == (cams.size[0] // 2, cams.size[1] // 2)
    assert torch.allclose(half.Ks[:, 0, 0], cams.Ks[:, 0, 0] / 2) and torch.equal(half.viewmats, cams.viewmats)
