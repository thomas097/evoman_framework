import numpy as np

# def mutate_offspring(offspring):
#     print(offspring.shape)
#     # Just add a smidge of random Gaussian noise.
#     noise = np.random.normal(0, 1, offspring.shape)
#     return offspring + noise

def mutate_offspring(offspring):
    # Add a little bit of self-adaptive noise (imperfect copy)
    if np.random.binomial(1, .1) == 1:
        mutated_sigma = np.random.normal(1, .25)
        noise = np.random.normal(0, mutated_sigma, offspring.shape)
        return offspring + noise*np.random.binomial(1, .15, offspring.shape)
    # vars gets 0 (Quasi-Deletion)
    if np.random.binomial(1, .05) == 1:
        return offspring*np.random.binomial(1, .85, offspring.shape)
    # Swap Coefficient
    if np.random.binomial(1, .05) == 1:
        return offspring*(np.random.binomial(1, .85, offspring.shape)*2-1)
    # vars gets 1
    if np.random.binomial(1, .05) == 1:
        return offspring**np.random.binomial(1, .85, offspring.shape)
    return offspring