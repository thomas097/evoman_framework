import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from tqdm import tqdm

import os
import numpy as np



def init_population(pop_size, num_vars):
    # Init population of uniformly sampled networks.
    pop = np.random.uniform(-1, 1, (pop_size, num_vars))
    return pop


def eval_population(pop, enemies=[1], trials=1):
    fitnesses = []
    # Evaluate each genome in the population.
    for genome in tqdm(pop):
        
        # Compute fitness as the average over enemy group.
        fitness = np.mean([eval_individual(genome, e, trials)[0] for e in enemies]) # [0] as [1] is gain
        fitnesses.append(fitness)
        
    return np.array(fitnesses)

def eval_individual(genome, enemy, trials):
    # Set up environment.
    env = Environment(experiment_name=None,
                      enemies=[enemy],
                      player_controller=player_controller(10), # magic number = hidden neurons
                      contacthurt='player',
                      speed='fastest',
                      logs = "off",
                      randomini='yes',
                      level=2)
    
    # Run enemy in environment and compute fitness/gain.
    fitness, gain = 0, 0
    for _ in range(trials):
        trial_fitness, player_en, enemy_en, _ = env.play(pcont=genome)
        trial_gain = player_en - enemy_en
        
        fitness += trial_fitness
        gain += trial_gain
    
    return fitness / trials, gain / trials

def parent_selection(population, fitnesses, num_parents):
    # Fitness proportional selection + windowing
    fitnesses = np.copy(fitnesses) - np.min(fitnesses)

#    selected_parents_inx = random_proportinal_parent_selection(population, fitnesses, num_parents)
    mating_pool_inx_inx = parent_selection_tournament(population, fitnesses, num_parents)
    mating_pool = population[mating_pool_inx_inx]
    return mating_pool

# proportional selection + windowing
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
    # Copy just in case....
    parent = np.copy(parents)
    
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


def fight_to_the_death(pop, pop_fitnesses, offspring, offspring_fitnesses):
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
    


if __name__ == "__main__":
    # EA params
    POP_SIZE = 100
    GENS = 2
    TRIALS = 1
    NUM_PARENTS = 15
    NUM_OFFSPRING = 15
    ENEMIES = [1, 2, 3]
    SHOW = False

    # Network params
    NUM_INPUTS = 20
    NUM_HIDDEN = 10
    NUM_OUTPUTS = 5
    NUM_VARS = (NUM_INPUTS + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUTPUTS

    # Do not show screen
    if not SHOW:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    ### Run evolution!

    # Init and evaluate population.
    pop = init_population(POP_SIZE, NUM_VARS)
    pop_fitnesses = eval_population(pop, ENEMIES, trials=TRIALS)
    
    for gen in range(GENS):
        # Select parents
        parents = parent_selection(pop, pop_fitnesses, NUM_PARENTS)

        # Recombine parents -> offspring
        offspring = recombine_parents(parents, NUM_OFFSPRING)

        # Mutate offspring
        offspring = mutate_offspring(offspring)

        # Evaluate kiddos.
        offspring_fitnesses = eval_population(offspring, ENEMIES, trials=TRIALS)

        # Survival selection
        pop, pop_fitnesses = fight_to_the_death(pop, pop_fitnesses, offspring, offspring_fitnesses)
        print("Stats: MEAN={}, MAX={}".format(np.mean(pop_fitnesses), np.max(pop_fitnesses)))
        

        
