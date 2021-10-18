
"""
    Filename:    generalist_boxplots.py
    Author(s):   Thomas Bellucci, Svetlana Codrean
    Assignment:  Task 2 - Evolutionary Computing
    Description: Creates the boxplots for the final report showing
                 gain scores for the best found player in each run
                 w.r.t the enemy group. It also performs the t-test 
                 to determine the significance of the performance 
                 differences found between the algorithms.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from evaluation import eval_individual
from demo_controller import player_controller


if __name__ == "__main__":
    TRAIN_ENEMIES = [[1, 5, 7],
                     [2, 3, 4]]
    TEST_ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]
    FITNESS_TYPES = [1, 2]
    RUNS = 10
    REPEATS = 5

    for k, train_enemies in enumerate(TRAIN_ENEMIES):
        print("Enemy group:", train_enemies)

        run_gains = np.zeros((len(FITNESS_TYPES), RUNS), dtype=float)
        
        for i, fitness_type in enumerate(FITNESS_TYPES):

            # Loop through runs
            for j, run in enumerate(range(RUNS)):

                filename = "solutions/best_run-{}_enemies-{}_fitness-{}.txt".format(run, train_enemies, fitness_type)
                ind = np.loadtxt(filename)

                gains = []
                for enemy in TEST_ENEMIES:
                    for _ in range(REPEATS):
                        _, gain = eval_individual(ind, fitness_type, enemy)
                        gains.append(gain * len(TEST_ENEMIES))
                        
                avg_gain = np.mean(gains)
                run_gains[i, j] = avg_gain
                print(f'run {run}, enemies {train_enemies}, fitness {fitness_type}, mean gain {avg_gain}')
         

        # Perform statistical Welch's t-test
        T, p = ttest_ind(run_gains[0], run_gains[1])
        print("enemies", train_enemies, "p =", p, "T =", T)

        # Format plots
        plt.subplot(1, 2, k+1)
        if k == 0:
            plt.ylabel("Gain")
        plt.boxplot(run_gains.T, widths=(0.75, 0.75))
        plt.xticks([1, 2], ["Eq. (1)", "Eq. (2)"])
        plt.ylim((-400, 160))
        plt.title("Training enemies {}".format(tuple(train_enemies)))
    plt.show()

