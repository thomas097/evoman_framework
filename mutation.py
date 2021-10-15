import numpy as np

# def mutate_offspring(offspring):
#     print(offspring.shape)
#     # Just add a smidge of random Gaussian noise.
#     noise = np.random.normal(0, 1, offspring.shape)
#     return offspring + noise
def mutate_offsprings(offsprings, prob = 0.2, self_adapt_sigma=1):
    
    for offspring in offsprings:
        val = np.random.uniform()
        if val <= prob:
            offspring = mutate_offspring(offspring, self_adapt_sigma)
    return offsprings

def mutate_offspring(offspring, sigma_loc=-1, tau=1, eps=0.1, self_adapt_sigma=1, sigma=1):
    # mutate sigma first
    if self_adapt_sigma:
        offspring[sigma_loc] += np.random.normal(0, tau, size=offspring[sigma_loc].shape)
        #offspring[sigma_loc] += np.random.normal(0, np.abs(offspring[sigma_loc]))

        # Boundary rule (i.e. sigma cannot be lower than eps; 1e10 is very big)
        offspring[sigma_loc] = np.clip(offspring[sigma_loc], eps, 1e10)
    mutated_sigma = offspring[sigma_loc] if self_adapt_sigma else sigma
    
    # Add a little bit of self-adaptive noise (imperfect copy)
    if np.random.binomial(1, .1) == 1:
        noise = np.random.normal(0, mutated_sigma, offspring.shape)
        offspring = offspring + noise*np.random.binomial(1, .15, offspring.shape)
        if self_adapt_sigma:
            offspring[sigma_loc] = mutated_sigma
        return offspring
    
    # vars gets 0
    if np.random.binomial(1, .05) == 1:
        return offspring*np.random.binomial(1, .85, offspring.shape)
    
    # Swap Coefficient
    if np.random.binomial(1, .05) == 1:
        return offspring*(np.random.binomial(1, .85, offspring.shape)*2-1)

    # vars gets 1
    if np.random.binomial(1, .05) == 1:
        return offspring**np.random.binomial(1, .85, offspring.shape)
    return offspring


mutation_rate = lambda gen, mut_init: mut_init*(1/gen**.25)