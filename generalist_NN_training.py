import sys, os
sys.path.insert(0, 'evoman')
os.environ["SDL_VIDEODRIVER"] = "dummy"

from environment import Environment
from demo_controller import player_controller
from tqdm import tqdm
import numpy as np
import argparse

# Import EA components
from evaluation import eval_population
from parent_selection2 import parent_selection
from recombination import uniform_crossover, single_point_crossover, multi_point_crossover
from mutation import mutate_offspring
from survivor_selection import survivor_selection
from logger import Logger


if __name__ == "__main__":
    # Parse arguments (if any are given)
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', help='number of runs per enemy', default=1, type = int)
    parser.add_argument('--generations', help = 'number of generations EA will run', default=100,type=int)
    parser.add_argument('--fitness', help = 'which fitness function to use (1: without time. 2: with time)', default=1, type=int)
    parser.add_argument('--population_size', help='population in each generation', default=100,type=int)
    parser.add_argument('--parents', help='number of parents for reproduction', default=20,type=int)
    parser.add_argument('--offsprings', help='number of offsprings', default=20,type=int)
    parser.add_argument('--enemies', help = 'comma separated types of enemies', default='1,2,3')
    parser.add_argument('--trials', help='number of trials', default=4,type=int)
    parser.add_argument('--self_adapt_sigma', help="allow self adaption of noise sigma", default=1, type=int)
    args = parser.parse_args()

    
    RUNS = args.runs
    FITNESS = args.fitness
    POP_SIZE = args.population_size
    GENS = args.generations
    TRIALS = args.trials
    NUM_PARENTS = args.parents
    NUM_OFFSPRING = args.offsprings
    ENEMIES = [int(i) for i in str(args.enemies).split(',')]
    SELF_ADAPT = int(args.self_adapt_sigma)

    # These are fixed by the assignment
    MIN_INIT = -1
    MAX_INIT = 1
    NUM_INPUTS = 20
    NUM_HIDDEN = 10
    NUM_OUTPUTS = 5
    NUM_VARS = (NUM_INPUTS + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUTPUTS + SELF_ADAPT

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
            off = uniform_crossover(par, NUM_OFFSPRING)
            off = mutate_offspring(off)
            off_fitnesses = eval_population(off, FITNESS, ENEMIES, num_trials=TRIALS)

            # Replacement
            pop, pop_fitnesses = survivor_selection(pop, pop_fitnesses, off, off_fitnesses)
            logger.log(pop_fitnesses)

        # Write best solution to file.
        logger.save_best(pop, pop_fitnesses)
