import sys, os
sys.path.insert(0, 'evoman')

import numpy as np


# selection methods with windowing: random, tournament of fitness sharing
def parent_selection(population, fitnesses, num_parents):
    # Windowing
    fitnesses = np.copy(fitnesses) - np.min(fitnesses)

    # Fitness sharing
    mating_pool_inx_inx = parent_selection_fitness_sharing(population, fitnesses, num_parents)
    return population[mating_pool_inx_inx]

# proportional random selection + windowing
def random_proportinal_parent_selection(pop, fitnesses, num_parents):
    fitnesses = np.copy(fitnesses) - np.min(fitnesses)
    pvals = fitnesses / np.sum(fitnesses)
    mating_pool_inx = np.random.choice(np.arange(pop.shape[0]), size=num_parents, p=pvals)
    return mating_pool_inx

# Deterministic K-Way tournament selection:
# 1) random select K individuals from the population at random
# 2) select the best out of these to become a parent.
# 3) repeat for selecting the next parent.
# Tournament Selection is also extremely popular in literature as it can even work with negative fitness values.
def parent_selection_tournament(population, fitnesses, num_parents, K=3):
    select_index = np.random.choice(len(population))
    mating_pool_inx = []
    for i in 0, num_parents-1 :
        for ix in np.random.randint(0, len(pop), K-1):
            if fitnesses[ix] < fitnesses[select_index]:
                select_index = ix
        mating_pool_inx.append(select_index)
    return mating_pool_inx

# Fitness sharing parent selection:
# Modern genetic algorithms usually devote a lot of effort to maintaining the diversity
# of the population to prevent premature convergence. One technique for that is fitness sharing.
# The inclusion of the fitness sharing technique in the evolutionary algorithm allows the extent to which
# the canonical genetic code is in an area corresponding to a deep local minimum to be easily determined,
# even in the high dimensional spaces considered.
def parent_selection_fitness_sharing(population, fitnesses, num_parents):
    pop_size = population.shape[0]

    mating_pool_inx = []
    for i in 0, num_parents - 1:
        candidate_size = 2;
        candidate_A_inx = np.random.randint(pop_size - 1)
        candidate_B_inx = np.random.randint(pop_size - 1)
        candidates_inx = [candidate_A_inx, candidate_B_inx]
        
        distances = np.zeros((candidate_size,), dtype=np.float64)
        for e, i in enumerate(candidates_inx):
            distances[e] = niches_count(i, fitnesses)
            
        # to maintain good diversity, best to choose the individual with smaller niche count.
        item_index = np.where(distances == distances.min())[0][0]
        candidate_index = candidates_inx[item_index]
        mating_pool_inx.append(candidate_index)
    return mating_pool_inx


def niches_count(candidate_index, fitnesses,  niche_radius = 1):
    niche_count = 0
    for ind in range(len(fitnesses) - 1):
        distance = np.linalg.norm(fitnesses[candidate_index] - fitnesses[ind])
        if distance <= niche_radius:
            sharing_func = 1.0 - (distance / niche_radius) # Linear
        else:
            sharing_func = 0
        niche_count = niche_count + sharing_func
    return niche_count
