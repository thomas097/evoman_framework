import numpy as np

def uniform_crossover(parents, num_offspring):

    """
    flip a coin for each chromosome to decide whether or not it’ll be included in the off-spring
    """
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


def single_point_crossover(parents, num_offspring):

    """
    a random crossover point is selected and the tails of its two parents are swapped to get new off-springs.
    """
    # Copy parents just in case....
    parents = np.copy(parents)

    offspring = []
    for _ in range(num_offspring // 2 + 1):
        # Select two parents randomly from parent pool
        np.random.shuffle(parents)
        p0 = parents[0]
        p1 = parents[1]

        parent_len = len(p0)

        #determine crossover point
        crossover_point =np.random.randint(0,parent_len)

     

        # Create two kiddos. 2nd kid will have inverse parental genes to 1st kid
        off0 = np.append(p0[0:crossover_point], p1[crossover_point:parent_len])
        off1 = np.append(p1[0:crossover_point], p0[crossover_point:parent_len])
        
        offspring.append(off0)
        offspring.append(off1)

    # Really make sure its not more than num_offspring.
    return np.array(offspring)[:num_offspring]


def multi_point_crossover(parents, num_offspring):

    """
    alternating segments are swapped to get new off-springs.
    """
    # Copy parents just in case....
    parents = np.copy(parents)

    offspring = []
    for _ in range(num_offspring // 2 + 1):
        # Select two parents randomly from parent pool
        np.random.shuffle(parents)
        p0 = parents[0]
        p1 = parents[1]

        parent_len = len(p0)
        parent_len_half = int(parent_len/2)

        #determine crossover points
        crossover_point_1 =np.random.randint(0,int(parent_len_half))
        crossover_point_2 =np.random.randint(int(parent_len_half),parent_len)
     

        # Create two kiddos. 2nd kid will have inverse parental genes to 1st kid
        off0 = np.hstack((p0[0:crossover_point_1], p1[crossover_point_1:crossover_point_2], p0[crossover_point_2: parent_len]))
        off1 = np.hstack((p1[0:crossover_point_1], p0[crossover_point_1:crossover_point_2], p1[crossover_point_2: parent_len]))
        
        offspring.append(off0)
        offspring.append(off1)

    # Really make sure its not more than num_offspring.
    return np.array(offspring)[:num_offspring]