import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from tqdm import tqdm

import os
import numpy as np


# Init population of uniformly sampled networks.
def init_population(pop_size, num_vars):
    pop = np.random.uniform(-1, 1, (pop_size, num_vars))
    return pop


# Evaluates each genome in the population.
def eval_population(pop, fitness_func, enemies, trials=1):
    pop_fitnesses = []
    for ind in tqdm(pop):

        # Average fitness over enemies.
        ind_fitness = 0
        for enemy in enemies:
            ind_fitness += eval_individual(ind, fitness_func, enemy, trials)[0]
        pop_fitnesses.append(ind_fitness / len(enemies))

    return np.array(pop_fitnesses)


def eval_individual(genome, fitness_func, enemy, trials):
    # Set up environment.
    env = Environment(experiment_name=None,
                      enemies=[enemy],
                      player_controller=player_controller(_n_hidden=10),
                      contacthurt='player',
                      speed='fastest',
                      logs="off",
                      randomini='yes',
                      level=2)

    fitnesses, gains = [], []
    for _ in range(trials):
        fitness, player_en, enemy_en, _ = env.play(pcont=genome)

        # Original fitness function
        if fitness_func == 1:
            fitnesses.append(fitness)

        # Alternative fitness function
        elif fitness_func == 2:
            fitnesses.append(0)  # TODO: implement other fitness function

        gains.append(player_en - enemy_en)

    return np.mean(fitnesses), np.mean(gains)


# selection methods with windowing: random, tournament of fitness sharing
def parent_selection(population, fitnesses, num_parents):
    # Windowing
    fitnesses = np.copy(fitnesses) - np.min(fitnesses)

    #mating_pool_inx_inx = random_proportinal_parent_selection(population, fitnesses, num_parents)
    mating_pool_inx_inx = parent_selection_tournament(population, fitnesses, num_parents)
    mating_pool = population[mating_pool_inx_inx]
    return mating_pool

# proportional random selection + windowing
def random_proportinal_parent_selection(pop, fitnesses, num_parents):
    fitnesses = np.copy(fitnesses) - np.min(fitnesses)
    pvals = fitnesses / np.sum(fitnesses)
    mating_pool_inx = np.random.choice(np.arange(pop.shape[0]), size=num_parents, p=pvals)
    return mating_pool_inx

# K-Way tournament selection:
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


def mutate_offspring(offspring):
    # Just add a smidge of random Gaussian noise.
    noise = np.random.normal(0, 1, offspring.shape)
    return offspring + 0.2 * noise


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

        with open(self.stats_fname + ".csv", "a") as f:
            f.write("{},{}\n".format(mean, _max))

        print("Stats: MEAN={} MAX={}".format(mean, _max))

    def save_best(self, pop, pop_fitnesses):
        # Save best solution from population.
        i = np.argmax(pop_fitnesses)
        solution = pop[i]

        solution_fname = self.stats_fname.replace("stats", "best")
        np.savetxt(solution_fname + ".txt", solution)


if __name__ == "__main__":
    # Program params
    RUNS = 1
    FITNESS = 1  # or 2 (Daniyal)
    SHOW = False

    # EA params
    POP_SIZE = 10
    GENS = 1
    TRIALS = 1
    NUM_PARENTS = 15
    NUM_OFFSPRING = 15
    ENEMIES = [1, 2, 3]

    # Network params
    NUM_INPUTS = 20
    NUM_HIDDEN = 10
    NUM_OUTPUTS = 5
    NUM_VARS = (NUM_INPUTS + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUTPUTS

    # Do not show screen
    if not SHOW:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    ###################
    ### Run evolution!
    ###################

    for run in range(RUNS):

        # Init stats logger
        logger = Logger(run, ENEMIES, FITNESS)

        # Set up and evaluate initial population
        pop = init_population(POP_SIZE, NUM_VARS)
        pop_fitnesses = eval_population(pop, FITNESS, ENEMIES, trials=TRIALS)
        logger.log(pop_fitnesses)

        for gen in range(GENS):
            # Evolutionary cycle
            parents = parent_selection(pop, pop_fitnesses, NUM_PARENTS)
            offspring = recombine_parents(parents, NUM_OFFSPRING)
            offspring = mutate_offspring(offspring)
            offspring_fitnesses = eval_population(offspring, FITNESS, ENEMIES, trials=TRIALS)
            pop, pop_fitnesses = survivor_selection(pop, pop_fitnesses, offspring, offspring_fitnesses)

            logger.log(pop_fitnesses)

        # Write best solution to file.
        logger.save_best(pop, pop_fitnesses)



