"""NSGA-II, MAP-Elites and CMA-ES on toy problems with known answers (CPU)."""

import numpy as np

from easplatting.cmaes import cma_es
from easplatting.map_elites import map_elites
from easplatting.nsga2 import (
    crowding_distance,
    front_2d,
    hypervolume,
    non_dominated_sort,
    nsga2,
)


def test_hypervolume_and_fronts():
    reference = np.array([1.0, 1.0])
    assert np.isclose(hypervolume(np.array([[0.2, 0.2]]), reference), 0.64)
    two = np.array([[0.2, 0.6], [0.6, 0.2], [0.7, 0.7]])  # last point is dominated
    assert np.isclose(hypervolume(two, reference), 0.48)
    assert hypervolume(np.array([[1.5, 0.1]]), reference) == 0.0  # outside the box
    fronts = non_dominated_sort(two)
    assert sorted(fronts[0]) == [0, 1] and fronts[1] == [2]
    assert list(front_2d(two)) == [0, 1]
    distance = crowding_distance(np.array([[0.0, 1.0], [0.4, 0.5], [1.0, 0.0]]))
    assert np.isinf(distance[0]) and np.isinf(distance[2]) and distance[1] == 2.0


def zdt1(genome):
    g = 1 + 9 * genome[1]
    return np.array([genome[0], g * (1 - np.sqrt(genome[0] / g))])


def test_nsga2_approaches_the_true_front():
    np.random.seed(0)
    reference = (1.1, 1.1)
    hv, genomes, objectives = nsga2(
        objective_function=zdt1,
        total_generations=40,
        population_size=20,
        genome_length=2,
        mutation_size=0.1,
        reference=reference,
    )
    x = np.linspace(0, 1, 200)
    best = hypervolume(np.stack([x, 1 - np.sqrt(x)], axis=1), np.array(reference))
    assert hv.shape == (40,) and hv[-1] > hv[0]
    assert hv[-1] > 0.9 * best  # within 10% of the hypervolume of the true front
    assert genomes.shape == (20, 2) and ((genomes >= 0) & (genomes <= 1)).all()


def test_map_elites_fills_the_grid():
    np.random.seed(0)
    toy = lambda genome: (1 - (genome[0] - 0.7) ** 2, genome[:2])
    coverage, qd_score, fitness, genomes = map_elites(
        fitness_function=toy,
        total_evaluations=600,
        num_initial=30,
        genome_length=3,
        mutation_size=0.1,
        bins=(5, 5),
    )
    assert np.all(np.diff(coverage) >= 0) and coverage[-1] > 0.9
    assert np.all(np.diff(qd_score) >= -1e-9)  # elites only ever get better
    assert fitness.shape == (5, 5) and genomes.shape == (5, 5, 3)
    assert fitness[3, 2] > 0.99  # the cell around genome[0] = 0.7 holds a near perfect elite


def test_cma_es_finds_the_target():
    np.random.seed(0)
    target = np.array([0.7, 0.2, 0.9, 0.4])
    toy = lambda genome: (-np.sum(np.square(genome - target)), 0.5, 40.0)
    fitness, solutions, kept, psnr = cma_es(
        fitness_function=toy,
        total_generations=40,
        population_size=16,
        genome_length=4,
    )
    assert np.all(np.diff(fitness) >= 0) and fitness[-1] > -1e-3
    assert np.allclose(solutions[-1], target, atol=0.05)


# ---- importance modes, coverage views, objective function, sweeps, tau, front summary

import json

import torch
from PIL import Image

from conftest import H, W
from easplatting.cameras import Cameras
from easplatting.experiment import (
    Log,
    OFF,
    importance_scores,
    run_sweep,
    tau_from_loss,
)
from easplatting.fitness import make_objective_function
from easplatting.front import ObjectiveLog, summarize_run
from easplatting.gs_loader import activate
from easplatting.plot_front import publish_front
from easplatting.render import (
    coverage_indices,
    gaussian_contributions,
    per_view_contributions,
    render_views,
)
from easplatting.thresholds import gene_stats, slider_bounds


def toy_cameras(count=2):
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    viewmats = torch.eye(4).repeat(count, 1, 1)
    viewmats[:, 2, 3] = 3.0
    viewmats[:, 0, 3] = torch.linspace(0, 0.5, count)  # cameras slide along x
    return Cameras(
        names=[f"v{i}" for i in range(count)],
        viewmats=viewmats,
        Ks=K.expand(count, 3, 3).clone(),
        widths=[W] * count,
        heights=[H] * count,
    )


def test_per_view_contributions_match_the_fake_rasterizer(raw_scene):
    scene = activate(raw_scene)
    cams = toy_cameras(3)
    each = per_view_contributions(scene, cams)
    both = gaussian_contributions(scene, cams, batch_size=3)
    seen = each["visible_views"] > 0
    assert torch.allclose(each["blend_weight"], both["blend_weight"], atol=1e-5)
    assert torch.allclose(each["max_weight"], scene["opacities"] * seen, atol=1e-5)
    assert torch.allclose(each["max_density"], each["max_weight"] / 9, atol=1e-6)
    assert (each["max_weight"] <= each["blend_weight"] + 1e-6).all()


def test_importance_modes_cache_and_differ(raw_scene, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = activate(raw_scene)
    cams = toy_cameras(3)
    scores = {
        mode: importance_scores("toy", scene, cams, [0, 1, 2], mode)
        for mode in ["sum", "max", "density", "lg"]
    }
    assert all(len(score) == len(scene["opacities"]) for score in scores.values())
    assert (tmp_path / "results/cache/toy_importance.pt").exists()
    assert (tmp_path / "results/cache/toy_importance_max.pt").exists()
    assert not torch.allclose(scores["sum"], scores["max"])


def test_coverage_indices_spread_over_the_cameras():
    cams = toy_cameras(10)
    chosen = coverage_indices(cams, list(range(10)), 3)
    assert chosen == [0, 4, 9] or chosen == [0, 5, 9]  # both ends and the middle
    assert coverage_indices(cams, [2, 3], 5) == [2, 3]


def test_objective_function_reports_size_quality_and_share(raw_scene):
    scene = activate(raw_scene)
    n = len(scene["opacities"])
    ranks = torch.arange(n, dtype=torch.float32)  # importance grows with the index
    stats = gene_stats(scene, importance=ranks)
    b = slider_bounds(stats)
    bounds = [b["min_opacity"], b["max_scale"], b["max_anisotropy"]]
    cams = toy_cameras(2)
    objective = make_objective_function(scene, stats, bounds, cams, render_views(scene, cams))
    kept, mean, worst, share = objective(np.array([0.0, 1.0, 1.0, 0.0]))
    assert (kept, mean, worst, share) == (1.0, 60.0, 60.0, 0.0)  # nothing removed
    kept, mean, worst, share = objective(np.array([0.0, 1.0, 1.0, 0.2]))
    assert 0.7 < kept < 0.85 and share == 1.0 and worst <= mean <= 60.0  # importance only


def test_sweep_refinement_finds_the_feasible_edge():
    edge = 0.37
    toy = lambda genome: (
        (genome[3], 1 - genome[3], 45.0)  # pruning more is better while feasible
        if genome[3] <= edge
        else (-1000.0, 1 - genome[3], 30.0)
    )
    coarse, fine = Log(toy, "sweep_importance", 0), Log(toy, "sweep_importance", 0)
    run_sweep(coarse, 4, 3, points=11)
    run_sweep(fine, 4, 3, points=11, refine=12)
    best = lambda log: max(r["fitness"] for r in log.rows)
    assert np.isclose(best(coarse), 0.3) and abs(best(fine) - edge) < 1e-3
    assert len(fine.rows) == len(coarse.rows) + 12 and fine.rows[0]["genome"][:3] == OFF[:3]


def test_tau_from_allowed_loss(tmp_path):
    photo = np.random.RandomState(0).rand(8, 8, 3).astype(np.float32) * 0.8 + 0.1
    paths = []
    for i in range(2):
        paths.append(tmp_path / f"{i}.png")
        Image.fromarray((photo * 255).round().astype(np.uint8)).save(paths[-1])
    cams = toy_cameras(2)
    cams.widths, cams.heights, cams.image_paths = [8, 8], [8, 8], [str(p) for p in paths]
    saved = torch.from_numpy(np.asarray(Image.open(paths[0])) / 255.0).float()
    baselines = (saved + 0.01).expand(2, 8, 8, 3).clone()  # model error of 1% = 40 dB
    tau = tau_from_loss(cams, baselines, allowed_loss=0.5)
    assert np.isclose(tau, 40 + 9.136 + 2, atol=0.05)  # G0 - 10 log10(10^(L/10) - 1) + 2


def test_front_summary_and_figure(tmp_path):
    rows, runs = [], []
    for method in ["nsga2", "map_elites", "random", "sweep"]:
        toy = lambda g: (g[0], 20 + 40 * g[0] - 5 * g[1], 20 + 40 * g[0], g[1])
        log = ObjectiveLog(toy, method, 0)  # bigger scenes look better: a real trade-off
        for g in np.random.RandomState(1).rand(40, 4):
            log(g)
        result = summarize_run(log.rows, "worst")
        result.update(method=method, seed=0, seconds=1.0)
        if method == "map_elites":
            result.update(archive=np.where(np.random.rand(10, 5) > 0.5, 40.0, np.nan).tolist())
        runs.append(result)
        rows += log.rows
    assert 0 < runs[0]["hypervolume"] < 1 and len(runs[0]["front"]) >= 2
    assert runs[0]["hypervolume_curve"][-1] == runs[0]["hypervolume"]
    held = {r["method"]: [dict(kept=p["kept"], fit_mean=p["mean"], fit_worst=p["worst"], test_mean=p["mean"] - 1, test_worst=p["worst"] - 2, genome=p["genome"]) for p in r["front"]] for r in runs}
    run = tmp_path / "results" / "toy_front_v32"
    run.mkdir(parents=True)
    args = dict(scene="toy", views=32, quality="worst", importance="sum")
    (run / "summary.json").write_text(json.dumps(dict(args=args, runs=runs, held_out=held)))
    png = publish_front(run, tmp_path / "figures")
    assert png == tmp_path / "figures" / "runs" / "toy_front_v32" / "front.png"
    assert png.stat().st_size > 10_000
