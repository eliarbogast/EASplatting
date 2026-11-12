"""ea.py — minimal evolutionary algorithm (mutation and tournament selection, no crossover)."""

import copy

import numpy as np

from easplatting.fitness import Individual


def evolutionary_algorithm(fitness_function=None, total_generations=100, num_parents=10, num_children=10, genome_length=3, mutation_size=0.1, tournament_size=4, num_tournament_winners=2):
    """ Evolutionary Algorithm (genes are fractions in [0, 1])

        parameters:
        fitness_function: (callable) genome -> (fitness, fraction of Gaussians kept, PSNR)
        total_generations: (int) number of total iterations for stopping condition
        num_parents: (int) the number of parents we downselect to at each generation (mu)
        num_children: (int) the number of children we balloon to each generation (lambda)
        genome_length: (int) length of the genome to be evolved
        mutation_size: (float) each gene gets (np.random.rand()*2-1)*mutation_size added
        tournament_size: (int) number of individuals competing in each tournament
        num_tournament_winners: (int) number of individuals selected as future parents from each tournament

        returns:
        fitness_over_time, solutions_over_time, kept_over_time, psnr_over_time, diversity_over_time:
        (numpy arrays) best-so-far fitness, genome, kept fraction and PSNR, and population diversity at each generation
    """

    # initialize record keeping
    solution = None # best individual so far
    solution_fitness = -99999999
    fitness_over_time = np.zeros(total_generations)
    solutions_over_time = np.zeros((total_generations, genome_length))
    kept_over_time = np.zeros(total_generations)
    psnr_over_time = np.zeros(total_generations)
    diversity_over_time = np.zeros(total_generations)

    # the initialization procedure
    population = []
    for i in range(num_parents):
        population.append(Individual(fitness_function, genome_length))

    # get population fitness
    for i in range(len(population)):
        population[i].eval_fitness()

    for generation_num in range(total_generations):

        # the modification procedure
        new_children = []
        while len(new_children) < num_children:

            # inheritance
            parent1 = np.random.choice(population)
            child1 = copy.deepcopy(parent1)

            # mutation
            child1.genome = np.clip(child1.genome + (np.random.rand(genome_length)*2-1)*mutation_size, 0, 1)
            new_children.append(child1)

        # the assessment procedure
        for i in range(len(new_children)):
            new_children[i].eval_fitness()

        # selection procedure
        population += new_children
        population = sorted(population, key=lambda individual: individual.fitness, reverse = True)

        # tournament selection
        new_population = []
        new_population.append(population[0])
        while len(new_population) < num_parents:
            tournament = np.random.choice(population, size = tournament_size, replace=False)
            tournament = sorted(tournament, key=lambda individual: individual.fitness, reverse = True)
            new_population.extend(tournament[:num_tournament_winners])
        population = new_population[:num_parents]

        # record keeping
        if population[0].fitness > solution_fitness:
            solution = population[0]
            solution_fitness = solution.fitness
        fitness_over_time[generation_num] = solution_fitness
        solutions_over_time[generation_num] = solution.genome
        kept_over_time[generation_num] = solution.kept
        psnr_over_time[generation_num] = solution.psnr

        # diversity - average std of each gene
        diversity_over_time[generation_num] = np.mean(np.std([individual.genome for individual in population], axis=0))

    return fitness_over_time, solutions_over_time, kept_over_time, psnr_over_time, diversity_over_time
