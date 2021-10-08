import numpy as np
from scipy.spatial import distance_matrix


def parent_selection(pop, fitnesses, num_parents):
    # Apply windowing
    fitnesses = np.copy(fitnesses) - np.min(fitnesses)

    # Fitness sharing
    fitnesses = fitness_sharing(pop, fitnesses)

    # Fitness proportional selection
    pvals = fitnesses / np.sum(fitnesses)
    idx = np.arange(len(pop))
    pool_idx = np.random.choice(idx, size=num_parents, p=pvals)
    return pop[pool_idx]

    
def fitness_sharing(pop, fitnesses):
    # Compute L1 distance between all population members i and j.
    dists = distance_matrix(pop, pop, p=1) # Genotypic
    #dists = distance_matrix(fitnesses.reshape(-1, 1), fitnesses.reshape(-1, 1), p=1) # Fitness shape
    
    # Share fitnesses
    fitnesses = [f / np.sum(sh(dists[i])) for i, f in enumerate(fitnesses)]
    return np.array(fitnesses)


def sh(dists, sigma_share=1, alpha=1):
    shares = 1 - ((dists / sigma_share) ** alpha)
    return shares * (dists <= sigma_share)
