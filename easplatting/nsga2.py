import numpy as np


def dominates(a, b):
    # a dominates b if it is no worse in every objective and better in one
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sort(objectives):
    # returns the fronts as lists of indices, best front first (all minimized)
    n = len(objectives)
    dominated_by = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1

    fronts = [[i for i in range(n) if domination_count[i] == 0]]
    while fronts[-1]:
        next_front = []
        for i in fronts[-1]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(objectives):
    # how isolated each point is within its front, the extremes are infinite
    n, m = objectives.shape
    distance = np.zeros(n)
    for k in range(m):
        order = np.argsort(objectives[:, k])
        distance[order[0]] = distance[order[-1]] = np.inf
        span = objectives[order[-1], k] - objectives[order[0], k]
        if span == 0:
            continue
        for i in range(1, n - 1):
            gap = objectives[order[i + 1], k] - objectives[order[i - 1], k]
            distance[order[i]] += gap / span
    return distance


def front_2d(objectives):
    # indices of the non-dominated points of a two objective set, sorted by the first
    objectives = np.asarray(objectives)
    order = np.lexsort((objectives[:, 1], objectives[:, 0]))
    front, best = [], np.inf
    for i in order:
        if objectives[i, 1] < best:
            front.append(i)
            best = objectives[i, 1]
    return np.array(front, dtype=int)


def hypervolume(objectives, reference):
    # area dominated by the points up to the reference point (two objectives, minimized)
    points = np.array([p for p in objectives if np.all(p < reference)])
    if len(points) == 0:
        return 0.0
    points = points[np.argsort(points[:, 0])]
    area, best = 0.0, reference[1]
    for i in range(len(points)):
        # strip from this point to the next one in the first objective
        best = min(best, points[i, 1])
        right = points[i + 1, 0] if i + 1 < len(points) else reference[0]
        area += (right - points[i, 0]) * (reference[1] - best)
    return area


def rank_and_crowding(objectives):
    ranks = np.zeros(len(objectives), dtype=int)
    crowding = np.zeros(len(objectives))
    for rank, front in enumerate(non_dominated_sort(objectives)):
        ranks[front] = rank
        crowding[front] = crowding_distance(objectives[front])
    return ranks, crowding


def nsga2(
    objective_function=None,
    total_generations=20,
    population_size=32,
    genome_length=4,
    mutation_size=0.1,
    reference=(1.0, 0.0),
    tournament_size=2,
):
    # objective_function(genome) -> objectives, all minimized

    # initialize record keeping
    hypervolume_over_time = np.zeros(total_generations)

    # the initialization procedure
    genomes = np.random.rand(population_size, genome_length)
    objectives = np.array([objective_function(genome) for genome in genomes])

    for generation_num in range(total_generations):
        # the modification procedure
        ranks, crowding = rank_and_crowding(objectives)
        children = []
        while len(children) < population_size:
            # tournament: lower rank wins, ties go to the less crowded point
            contenders = np.random.choice(
                len(genomes), size=tournament_size, replace=False
            )
            winner = min(contenders, key=lambda i: (ranks[i], -crowding[i]))

            # mutation
            child = np.clip(
                genomes[winner]
                + (np.random.rand(genome_length) * 2 - 1) * mutation_size,
                0,
                1,
            )
            children.append(child)

        # the assessment procedure
        children = np.array(children)
        child_objectives = np.array([objective_function(child) for child in children])

        # selection procedure: fill with whole fronts, split the last by crowding
        genomes = np.vstack([genomes, children])
        objectives = np.vstack([objectives, child_objectives])
        keep = []
        for front in non_dominated_sort(objectives):
            if len(keep) + len(front) <= population_size:
                keep.extend(front)
            else:
                distance = crowding_distance(objectives[front])
                order = np.argsort(-distance)
                keep.extend(np.array(front)[order][: population_size - len(keep)])
                break
        genomes, objectives = genomes[keep], objectives[keep]

        # record keeping
        hypervolume_over_time[generation_num] = hypervolume(objectives, reference)

    return hypervolume_over_time, genomes, objectives
