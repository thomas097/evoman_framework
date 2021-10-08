import numpy as np

def recombine_parents(parents, num_offspring):
    # Copy parents just in case....
    parents = np.copy(parents)

    offspring = []
    for _ in range(num_offspring // 2 + 1):
        # Select two parents randomly from parent pool
        np.random.shuffle(parents)
        p0 = parents[0]
        p1 = parents[1]

        # Determine which alleles come from which parent.
        m = np.uint8(np.random.random(p0.shape) < 0.5)

        # Create two kiddos.
        off0 = p0 * m + p1 * (1 - m)
        off1 = p1 * m + p0 * (1 - m)
        
        offspring.append(off0)
        offspring.append(off1)

    # Really make sure its not more than num_offspring.
    return np.array(offspring)[:num_offspring]
