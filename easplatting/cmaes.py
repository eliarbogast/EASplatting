import numpy as np


def cma_es(
    fitness_function=None,
    total_generations=20,
    population_size=32,
    genome_length=4,
    step_size=0.3,
):
    # fitness_function(genome) -> (fitness, kept, psnr), maximized
    # genomes are clipped to [0, 1]
    n = genome_length
    mu = population_size // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1 / np.sum(weights**2)

    # adaptation constants
    cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    cs = (mu_eff + 2) / (n + mu_eff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    # the initialization procedure
    mean = np.full(n, 0.5)
    sigma = step_size
    covariance = np.eye(n)
    path_c = np.zeros(n)
    path_s = np.zeros(n)

    # initialize record keeping
    solution = None  # best genome so far
    solution_fitness = -99999999
    fitness_over_time = np.zeros(total_generations)
    solutions_over_time = np.zeros((total_generations, n))
    kept_over_time = np.zeros(total_generations)
    psnr_over_time = np.zeros(total_generations)
    kept, psnr = 1.0, 0.0

    for generation_num in range(total_generations):
        # the modification procedure: sample around the mean
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scales = np.sqrt(np.maximum(eigenvalues, 1e-20))
        steps = (np.random.randn(population_size, n) * scales) @ eigenvectors.T
        samples = np.clip(mean + sigma * steps, 0, 1)
        steps = (samples - mean) / sigma  # adapt to the steps that were evaluated

        # the assessment procedure
        results = [fitness_function(sample) for sample in samples]
        fitness = np.array([result[0] for result in results])

        # selection procedure: move toward the best mu samples
        order = np.argsort(-fitness)
        best_steps = steps[order[:mu]]
        mean_step = weights @ best_steps
        mean = np.clip(mean + sigma * mean_step, 0, 1)
        inverse_sqrt = eigenvectors @ np.diag(1 / scales) @ eigenvectors.T
        path_s = (1 - cs) * path_s + np.sqrt(cs * (2 - cs) * mu_eff) * (
            inverse_sqrt @ mean_step
        )
        decay = np.sqrt(1 - (1 - cs) ** (2 * (generation_num + 1)))
        h_sigma = np.linalg.norm(path_s) / decay / chi_n < 1.4 + 2 / (n + 1)
        scale_c = np.sqrt(cc * (2 - cc) * mu_eff)
        path_c = (1 - cc) * path_c + h_sigma * scale_c * mean_step
        rank_one = np.outer(path_c, path_c) + (1 - h_sigma) * cc * (2 - cc) * covariance
        rank_mu = (best_steps.T * weights) @ best_steps
        covariance = (1 - c1 - cmu) * covariance + c1 * rank_one + cmu * rank_mu
        sigma *= np.exp((cs / damps) * (np.linalg.norm(path_s) / chi_n - 1))

        # record keeping
        if fitness[order[0]] > solution_fitness:
            solution = samples[order[0]]
            solution_fitness = fitness[order[0]]
            kept, psnr = results[order[0]][1], results[order[0]][2]
        fitness_over_time[generation_num] = solution_fitness
        solutions_over_time[generation_num] = solution
        kept_over_time[generation_num] = kept
        psnr_over_time[generation_num] = psnr

    return fitness_over_time, solutions_over_time, kept_over_time, psnr_over_time
