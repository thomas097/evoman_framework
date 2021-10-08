import numpy as np

def mutate_offspring(offspring):
    # Just add a smidge of random Gaussian noise.
    noise = np.random.normal(0, 1, offspring.shape)
    return offspring + noise
