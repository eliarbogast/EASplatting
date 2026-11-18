import numpy as np


def map_elites(
    fitness_function=None,
    total_evaluations=672,
    num_initial=32,
    genome_length=4,
    mutation_size=0.1,
    bins=(10, 5),
):
    # fitness_function(genome) -> (fitness, descriptors), descriptors in [0, 1]
    bins = np.array(bins)

    # initialize record keeping: the best genome found in each cell of the grid
    archive_fitness = np.full(tuple(bins), -np.inf)
    archive_genomes = np.zeros(tuple(bins) + (genome_length,))
    coverage_over_time = np.zeros(total_evaluations)
    qd_score_over_time = np.zeros(total_evaluations)

    for evaluation_num in range(total_evaluations):
        # the modification procedure
        if evaluation_num < num_initial:
            genome = np.random.rand(genome_length)
        else:
            # pick a random elite and mutate it
            elites = np.argwhere(np.isfinite(archive_fitness))
            cell = tuple(elites[np.random.randint(len(elites))])
            genome = np.clip(
                archive_genomes[cell]
                + (np.random.rand(genome_length) * 2 - 1) * mutation_size,
                0,
                1,
            )

        # the assessment procedure
        fitness, descriptors = fitness_function(genome)

        # selection procedure: replace the elite of the cell if this one is better
        scaled = (np.clip(descriptors, 0, 1) * bins).astype(int)
        cell = tuple(np.minimum(scaled, bins - 1))
        if fitness > archive_fitness[cell]:
            archive_fitness[cell] = fitness
            archive_genomes[cell] = genome

        # record keeping
        filled = np.isfinite(archive_fitness)
        coverage_over_time[evaluation_num] = filled.mean()
        qd_score_over_time[evaluation_num] = archive_fitness[filled].sum()

    return coverage_over_time, qd_score_over_time, archive_fitness, archive_genomes
