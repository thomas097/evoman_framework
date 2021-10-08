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

        # Determine crossover point.
        xpoint = np.random.randint(low=1, high=p0.shape[0])

        # Create two kiddos.
        off0 = np.zeros(p0.shape, dtype=np.float64)
        off0[:xpoint] = p0[:xpoint]
        off0[xpoint:] = p1[xpoint:]
        offspring.append(off0)

        off1 = np.zeros(p0.shape, dtype=np.float64)
        off1[:xpoint] = p1[:xpoint]
        off1[xpoint:] = p0[xpoint:]
        offspring.append(off1)

    # Really make sure its not more than num_offspring.
    return np.array(offspring)[:num_offspring]
