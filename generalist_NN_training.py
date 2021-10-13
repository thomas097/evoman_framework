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
from recombination import uniform_crossover,single_point_crossover,multi_point_crossover
from mutation import mutate_offsprings
from survivor_selection import survivor_selection
from logger import Logger

crossover_dict = {1: uniform_crossover, 2:single_point_crossover, 3:multi_point_crossover}

if __name__ == "__main__":
    # Parse arguments (if any are given)
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', help='number of runs per enemy', default=10, type = int)
    parser.add_argument('--generations', help = 'number of generations EA will run', default=75,type=int)
    parser.add_argument('--fitness', help = 'which fitness function to use (1: without time. 2: with time)', default=2, type=int)
    parser.add_argument('--population_size', help='population in each generation', default=100,type=int)
    parser.add_argument('--parents', help='number of parents for reproduction', default=20,type=int)
    parser.add_argument('--offsprings', help='number of offsprings', default=20,type=int)
    parser.add_argument('--enemies', help = 'comma separated types of enemies', default='2,3,4')
    parser.add_argument('--min_init', help='minimum initializations', default=-1,type=int)
    parser.add_argument('--max_init', help='maximum initialization', default=1,type=int)
    parser.add_argument('--inputs', help='input size of NN', default=20,type=int)
    parser.add_argument('--hidden', help='hidden layer size of NN', default=10,type=int)
    parser.add_argument('--outputs', help='output size of NN', default=5,type=int)
    parser.add_argument('--crossover', help='crossover type (1: uniform, 2: singlepoint, 3: multipoint)', default=1,type=int)
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

    MIN_INIT = args.min_init
    MAX_INIT = args.max_init
    NUM_INPUTS = args.inputs
    NUM_HIDDEN = args.hidden
    NUM_OUTPUTS = args.outputs
    NUM_VARS = (NUM_INPUTS + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUTPUTS

    if args.self_adapt_sigma:
        NUM_VARS += 1

    CROSSOVER = args.crossover

    print(args)
    for run in range(RUNS):
        print(f'RUN: {run+1}')
        # Init stats logger
        logger = Logger(run, ENEMIES, FITNESS)

        # Initialize and evaluate initial population
        pop = np.random.uniform(MIN_INIT, MAX_INIT, (POP_SIZE, NUM_VARS))
        pop_fitnesses = eval_population(pop, FITNESS, ENEMIES, num_trials=TRIALS)
        logger.log(pop_fitnesses)

        for gen in range(GENS):
            print(f'RUN: {run+1}||GENERATION: {gen+1}')
            # print(gen)
            # Parent selection
            par = parent_selection(pop, pop_fitnesses, NUM_PARENTS)

            # Reproduction
            off = crossover_dict[CROSSOVER](par, NUM_OFFSPRING)
            # print(off.shape)
            off = mutate_offsprings(off)
            off_fitnesses = eval_population(off, FITNESS, ENEMIES, num_trials=TRIALS)

            # Replacement
            pop, pop_fitnesses = survivor_selection(pop, pop_fitnesses, off, off_fitnesses)
            logger.log(pop_fitnesses)

        # Write best solution to file.
        logger.save_best(pop, pop_fitnesses)
