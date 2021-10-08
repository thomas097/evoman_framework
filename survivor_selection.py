import numpy as np


def survivor_selection(pop, pop_fitnesses, offspring, offspring_fitnesses):
    # Combine offspring and old population
    total_pop = np.vstack([pop, offspring])
    total_fitnesses = np.hstack([pop_fitnesses, offspring_fitnesses])

    # Sort new population w.r.t fitness (highest fitness at index 0)
    i = total_fitnesses.argsort()[::-1]
    total_pop = total_pop[i]
    total_fitnesses = total_fitnesses[i]

    # Deterministic competition for survivl (mu + lambda) style.
    pop_size = pop.shape[0]
    return total_pop[:pop_size], total_fitnesses[:pop_size]
