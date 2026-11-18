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
from easplatting.thresholds import gene_stats, keep_mask, slider_bounds

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


def test_importance_gene(raw_scene):
    scene = activate(raw_scene)
    n = scene["means"].shape[0]
    stats = gene_stats(scene, importance=torch.arange(n, dtype=torch.float32).flip(0))   # Gaussian 0 is the most important
    assert stats["importance"][0] > stats["importance"][-1] and stats["importance"].min() == 0
    assert keep_mask(stats, 0.0, np.inf, np.inf).all()                                   # off by default
    mask = keep_mask(stats, 0.0, np.inf, np.inf, 0.5)
    assert abs(mask.sum().item() - n / 2) <= 1 and mask[0] and not mask[-1]              # removes the least important half
    assert decode([0.5, 0.5, 0.5, 0.04], BOUNDS)[3] == 0.0 and decode([0.5, 0.5, 0.5, 0.6], BOUNDS)[3] == 0.6
    assert len(decode([0.5, 0.5, 0.5], BOUNDS)) == 3


def test_experiment_sweep_and_log():
    from easplatting.experiment import Log, OFF, best_row, run_sweep
    fit = Log(lambda g: (-abs(g[3] - 0.4), 1 - g[3], 45.0), "sweep_importance", 0)
    run_sweep(fit, 4, 3, points=11)
    assert len(fit.rows) == 11 and all(r["genome"][:3] == OFF[:3] for r in fit.rows)      # other genes stay off
    assert best_row(fit.rows)["genome"][3] == 0.4 and fit.rows[0]["eval"] == 0


def test_fitness_worst_view_floor(raw_scene):
    scene = activate(raw_scene)
    stats = gene_stats(scene)
    b = slider_bounds(stats)
    bounds = [b["min_opacity"], b["max_scale"], b["max_anisotropy"]]
    cams = Cameras(names=["a", "b"], viewmats=torch.stack([torch.eye(4), torch.eye(4)]), Ks=torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]]).expand(2, 3, 3).clone(), widths=[W, W], heights=[H, H])
    cams.viewmats[:, 2, 3] = 3.0
    base = render_views(scene, cams)
    all_off = np.array([0.0, 1.0, 1.0])
    assert make_fitness_function(scene, stats, bounds, cams, base, 40.0, worst_tau=50.0)(all_off)[0] == 0.0   # PSNR 60 on both views
    assert make_fitness_function(scene, stats, bounds, cams, base, 40.0, worst_tau=70.0)(all_off)[0] == -1000.0 - 10.0


def test_plot_run_publish(tmp_path):
    import json
    from easplatting.plot_run import publish
    run = tmp_path / "results" / "toy_4g"
    run.mkdir(parents=True)
    rows, runs = [], []
    for method, seed in [("ga", 0), ("random", 0), ("sweep_importance", 0), ("sweep_opacity", 0)]:
        for i in range(20):
            k = 1 - i / 40
            rows.append(dict(method=method, seed=seed, eval=i, genome=[0.5] * 4, fitness=1 - k, kept=k, psnr=60 - 30 * (1 - k) * 2, ms=1.0))
        runs.append(dict(method=method, seed=seed, evals=20, seconds=1.0, feasible=True, genome=[0.5] * 4, kept=0.6, fit_psnr=41.0,
                         train_psnr=40.5, train_worst=30.0, test_psnr=39.5, test_worst=28.0))
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "summary.json").write_text(json.dumps(dict(args=dict(scene="bonsai", genes=4, tau=40.0, views=8, worst_tau=None), runs=runs)))
    png, compare = publish(run, tmp_path / "figures")
    assert png.exists() and compare.exists() and (tmp_path / "figures" / "runs" / "toy_4g.json").exists()
    assert (run / "run.png").stat().st_size > 10_000


def test_strip_worst_window_and_picks():
    from easplatting.strip import pick_genomes, worst_window
    err = torch.zeros(80, 120)
    err[50:60, 90:100] = 1.0                                              # the damaged spot
    y, x, h, w = worst_window(err, zoom=4)
    assert (h, w) == (20, 30) and y <= 50 and y + h >= 60 and x <= 90 and x + w >= 100
    runs = [dict(method="ga", feasible=True, kept=0.6), dict(method="ga", feasible=True, kept=0.5), dict(method="ga", feasible=False, kept=0.1),
            dict(method="random", feasible=True, kept=0.3), dict(method="sweep_opacity", feasible=True, kept=0.7)]
    picks = pick_genomes(runs)
    assert picks["ga"]["kept"] == 0.5 and "random" not in picks and list(picks) == ["ga", "sweep_opacity"]
