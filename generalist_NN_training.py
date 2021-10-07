import sys, os

sys.path.insert(0, 'evoman')
os.environ["SDL_VIDEODRIVER"] = "dummy"

from environment import Environment
from demo_controller import player_controller
from tqdm import tqdm
import numpy as np


##############################
## Initialization
##############################

# Init population of uniformly sampled networks.
def init_population(pop_size, num_vars):
    pop = np.random.uniform(-1, 1, (pop_size, num_vars))
    return pop


##############################
## Evaluation
##############################

# Evaluates each genome in the population.
def eval_population(pop, fitness_func, enemies, num_trials=1):
    pop_fitnesses = []
    for ind in tqdm(pop):

        # Average fitness over x trials for y enemies.
        ind_fitness = []
        for enemy in enemies:
            for _ in range(num_trials):
                ind_fitness.append( eval_individual(ind, fitness_func, enemy)[0] )
        pop_fitnesses.append(np.mean(ind_fitness))

    return np.array(pop_fitnesses)


def eval_individual(genome, fitness_func, enemy):
    # Set up environment for enemy.
    env = Environment(experiment_name=None,
                      enemies=[enemy],
                      player_controller=player_controller(_n_hidden=10),
                      contacthurt='player',
                      speed='fastest',
                      logs="off",
                      randomini='yes',
                      level=2)

    # Run game.
    game_fitness, player_en, enemy_en, _ = env.play(pcont=genome)
    gain = player_en - enemy_en

    # Original fitness function
    if fitness_func == 1:
        fitness = game_fitness

    # Alternative fitness function
    elif fitness_func == 2:
        fitness = 0 # TODO: Daniyal

    return fitness, gain


##############################
## Parent Selection
##############################

# selection methods with windowing: random, tournament of fitness sharing
def parent_selection(population, fitnesses, num_parents):
    # Windowing
    fitnesses = np.copy(fitnesses) - np.min(fitnesses)

    #mating_pool_inx_inx = random_proportinal_parent_selection(population, fitnesses, num_parents)
    #mating_pool_inx_inx = parent_selection_tournament(population, fitnesses, num_parents)
    mating_pool_inx_inx = parent_selection_fitness_sharing(population, fitnesses, num_parents)
    mating_pool = population[mating_pool_inx_inx]
    return mating_pool

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
    mating_pool_inx = []
    for i in 0, num_parents - 1 :
        candidate_size = 2;
        candidate_A_inx = np.random.randint(POP_SIZE - 1)
        candidate_B_inx = np.random.randint(POP_SIZE - 1)
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
    for ind in range(POP_SIZE - 1):
        distance = np.linalg.norm(fitnesses[candidate_index] - fitnesses[ind])
        if distance <= niche_radius:
            sharing_func = 1.0 - (distance / niche_radius) # Linear
        else:
            sharing_func = 0
        niche_count = niche_count + sharing_func
    return niche_count


##############################
## Recombination
##############################

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


##############################
## Mutation
##############################

def mutate_offspring(offspring):
    # Just add a smidge of random Gaussian noise.
    noise = np.random.normal(0, 1, offspring.shape)
    return offspring + noise


##############################
## Survivor Selection
##############################

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


##############################
## Utilities / Main loop
##############################

class Logger:
    """ Convenience class to log the mean and max fitness of
        genotypes in the population over the course of a run.
    """

    def __init__(self, run, enemies, fitness):
        # Set up file with header.
        self.stats_fname = "stats_run-{}_enemies-{}_fitness-{}".format(run, enemies, fitness)
        with open(self.stats_fname + ".csv", "w") as f:
            f.write("mean,max\n")

    def log(self, pop_fitnesses):
        # Print population statistics.
        mean = np.mean(pop_fitnesses)
        _max = np.max(pop_fitnesses)
        print("Stats: MEAN={} MAX={}".format(mean, _max))

        # Write stats to file.
        with open(self.stats_fname + ".csv", "a") as f:
            f.write("{},{}\n".format(mean, _max))

    def save_best(self, pop, pop_fitnesses):
        # Write best solution to file.
        solution = pop[np.argmax(pop_fitnesses)]
        solution_fname = self.stats_fname.replace("stats", "best")
        np.savetxt(solution_fname + ".txt", solution)


if __name__ == "__main__":
    # Program params
    RUNS = 10
    FITNESS = 1  # or 2

    # EA params
    POP_SIZE = 100
    GENS = 50
    TRIALS = 5
    NUM_PARENTS = 20
    NUM_OFFSPRING = 20
    ENEMIES = [1, 2, 3]

    # Network topology
    NUM_INPUTS = 20
    NUM_HIDDEN = 10
    NUM_OUTPUTS = 5
    NUM_VARS = (NUM_INPUTS + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUTPUTS

    ####################
    ### Run evolution!
    ####################

    # Do not show screen
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    for run in range(RUNS):

        # Init stats logger
        logger = Logger(run, ENEMIES, FITNESS)

        # Set up and evaluate initial population
        pop = init_population(POP_SIZE, NUM_VARS) 
        pop_fitnesses = eval_population(pop, FITNESS, ENEMIES, num_trials=TRIALS)
        logger.log(pop_fitnesses)

        for gen in range(GENS):
            # Parent selection
            par = parent_selection(pop, pop_fitnesses, NUM_PARENTS)

            # Reproduction
            off = recombine_parents(par, NUM_OFFSPRING)
            off = mutate_offspring(off)
            off_fitnesses = eval_population(off, FITNESS, ENEMIES, num_trials=TRIALS)

            # Replacement
            pop, pop_fitnesses = survivor_selection(pop, pop_fitnesses, off, off_fitnesses)
            logger.log(pop_fitnesses)

        # Write best solution to file.
        logger.save_best(pop, pop_fitnesses)



