import sys, os
sys.path.insert(0, 'evoman')
os.environ["SDL_VIDEODRIVER"] = "dummy"

from environment import Environment
from demo_controller import player_controller
from tqdm import tqdm
import numpy as np

# Import EA components
from evaluation import eval_population
from parent_selection2 import parent_selection
from recombination import recombine_parents
from mutation import mutate_offspring
from survivor_selection import survivor_selection
from logger import Logger



# Params
RUNS = 1
FITNESS = 1  # or 2

POP_SIZE = 100
GENS = 20
TRIALS = 5
NUM_PARENTS = 20
NUM_OFFSPRING = 20
ENEMIES = [1, 2, 3]

MIN_INIT = -1
MAX_INIT = 1
NUM_INPUTS = 20
NUM_HIDDEN = 10
NUM_OUTPUTS = 5
NUM_VARS = (NUM_INPUTS + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUTPUTS



if __name__ == "__main__":

    for run in range(RUNS):

        # Init stats logger
        logger = Logger(run, ENEMIES, FITNESS)

        # Initialize and evaluate initial population
        pop = np.random.uniform(MIN_INIT, MAX_INIT, (POP_SIZE, NUM_VARS))
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



