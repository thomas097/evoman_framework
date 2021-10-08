import numpy as np

def mutate_offspring(offspring):
    # Just add a smidge of random Gaussian noise.
    noise = np.random.normal(0, 2, offspring.shape)
    return offspring + noise
