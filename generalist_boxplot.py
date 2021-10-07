
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

from generalist_NN_training import eval_individual
from demo_controller import player_controller


if __name__ == "__main__":
    ENEMIES = [[1, 2, 3],
               [1, 2, 3]]
    FITNESS_TYPES = [1, 2]
    RUNS = 10
    REPEATS = 5

    for k, enemies in enumerate(ENEMIES):
        print("Enemy group", enemies)

        run_gains = np.zeros((len(FITNESS_TYPES), len(RUNS)), dtype=float)
        
        for i, fitness_type in enumerate(FITNESS_TYPES):

            # Loop through runs
            for j, run in enumerate(range(RUNS)):

                filename = "best_run-{}_enemies-{}_fitness-{}.txt".format(run, enemies, fitness_type)
                ind = np.loadtxt(filename)

                gains = []
                for enemy in enemies:
                    for _ in range(REPEATS):
                        _, gain = eval_individual(ind, fitness_type, enemy)
                        gains.append(gain)
                avg_gain = np.mean(gains)

                print(f'run {run}, enemies {enemies}, mean gain {avg_gain}')
                run_gains[i, j] = avg_gain

        # Perform statistical Welch's t-test
        T, p = ttest_ind(run_gains[0], run_gains[1], equal_var=False)
        print("enemies", enemies, "p =", p, "T =", T)

        # Format plots
        plt.subplot(1, 3, k+1)
        if k == 0:
            plt.ylabel("Gain")
        plt.boxplot(run_gains.T, widths=(0.75, 0.75))
        plt.xticks([1, 2], ["EA1", "EA2"])
        plt.ylim((-20, 110))
        plt.title("Enemies {}".format(enemy))
    plt.show()
    


