
"""
    Filename:    specialist_boxplots.py
    Author(s):   Thomas Bellucci, Svetlana Codrean
    Assignment:  Task 1 - Evolutionary Computing
    Description: Creates the boxplots for the final report showing
                 individual gain scores for the best found player
                 in each run w.r.t the enemy. It also performs the
                 Welch's t-test to determine the significance of
                 the performance differences found between the algorithms.
"""

import pickle
import neat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from specialist_neat_training import Individual, EvomanEnvironment


if __name__ == "__main__":
    ENEMIES = [2, 3, 7]
    INDIVIDUAL_TYPES = [1, 3]
    RUNS = list(range(1, 11))
    REPEATS = 5

    for k, enemy in enumerate(ENEMIES):
        print("Enemy", enemy)

        run_gains = np.zeros((len(INDIVIDUAL_TYPES), len(RUNS)), dtype=float)
        
        for i, ind_type in enumerate(INDIVIDUAL_TYPES):

            # Load configuration file.
            if ind_type == 1:
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     "neat.config")
            else:
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     "neat_fixed_topology.config")

            # Loop through runs
            for j, run in enumerate(RUNS):

                filename = "solutions/neat_best_run-{}_enemy-{}_ind-{}.pkl".format(run, enemy, ind_type)
                with open(filename, "rb") as f:
                    genome = pickle.load(f)

                # Run game
                env = EvomanEnvironment(enemy, run)

                gains = []
                for _ in range(REPEATS):
                    fitness, gain = env.evaluate_individual(genome, config, show=False)
                    gains.append(gain)
                avg_gain = np.mean(gains)

                print(f'run {run}, enemy {enemy}, mean gain {avg_gain}')
                run_gains[i, j] = avg_gain

        # Perform statistical Welch's t-test
        T, p = ttest_ind(run_gains[0], run_gains[1], equal_var=False)
        print("enemy", enemy, "p =", p, "T =", T)

        # Format plots
        plt.subplot(1, 3, k+1)
        if k == 0:
            plt.ylabel("Individual Gain")
        plt.boxplot(run_gains.T, widths=(0.75, 0.75))
        plt.xticks([1, 2], ["NEAT", "Fixed\nNEAT"])
        plt.ylim((-20, 110))
        plt.title("Enemy {}".format(enemy))
    plt.show()
    


