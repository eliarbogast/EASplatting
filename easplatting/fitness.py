# fitness func here
import numpy as np


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
    """ returns fitness_function(genome) -> (fitness, kept fraction, PSNR) """
    raise NotImplementedError
