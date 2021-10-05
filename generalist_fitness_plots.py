"""
    Filename:    specialist_fitness_plots.py
    Author(s):   Thomas Bellucci, Svetlana Codrean
    Assignment:  Task 1 - Evolutionary Computing
    Description: Creates the fitness curves for the final report
                 showing the mean and max fitness for the population
                 as a function of the generation. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ENEMIES = [2, 3, 7]
    INDIVIDUAL_TYPES = [1, 3]
    RUNS = list(range(1, 11))

    for k, enemy in enumerate(ENEMIES):

        mean_EA1 = []
        mean_EA3 = []
        max_EA1 = []
        max_EA3 = []

        # Load in mean and max stats for each run.
        for run in RUNS:
            for ind_type in INDIVIDUAL_TYPES:
                filename = "stats/neat_stats_run-{}_enemy-{}_ind-{}.csv".format(run, enemy, ind_type)
                df = pd.read_csv(filename)

                if ind_type == 1:
                    mean_EA1.append(df["mean"])
                    max_EA1.append(df["max"])
                    
                elif ind_type == 3:
                    mean_EA3.append(df["mean"])
                    max_EA3.append(df["max"])

        # Compute mean and stdev over runs of mean and max population stats.
        x = np.arange(1, 31)
        mean_mean_EA1 = np.mean(mean_EA1, axis=0)
        mean_mean_EA3 = np.mean(mean_EA3, axis=0)
        mean_max_EA1 = np.mean(max_EA1, axis=0)
        mean_max_EA3 = np.mean(max_EA3, axis=0)

        std_mean_EA1 = np.std(mean_EA1, axis=0)
        std_mean_EA3 = np.std(mean_EA3, axis=0)
        std_max_EA1 = np.std(max_EA1, axis=0)
        std_max_EA3 = np.std(max_EA3, axis=0)

        # Plot mean and stdev of mean population fitness.
        plt.subplot(1, 3, k+1)
        plt.plot(x, mean_mean_EA1, label="NEAT (mean)", c="C0")
        plt.fill_between(x, mean_mean_EA1 - std_mean_EA1, mean_mean_EA1 + std_mean_EA1, alpha=0.4, fc="C0")
        
        plt.plot(x, mean_mean_EA3, label="Fixed NEAT (mean)", c="C1")
        plt.fill_between(x, mean_mean_EA3 - std_mean_EA3, mean_mean_EA3 + std_mean_EA3, alpha=0.4, fc="C1")

        # Plot mean and stdev of mean population fitness.
        plt.subplot(1, 3, k+1)
        plt.plot(x, mean_max_EA1, label="NEAT (max)", c="C0", linestyle="--")
        plt.fill_between(x, mean_max_EA1 - std_max_EA1, mean_max_EA1 + std_max_EA1, alpha=0.4, fc="C0")
        
        plt.plot(x, mean_max_EA3, label="Fixed NEAT (max)", c="C1", linestyle="--")
        plt.fill_between(x, mean_max_EA3 - std_max_EA3, mean_max_EA3 + std_max_EA3, alpha=0.4, fc="C1")

        plt.ylim((0, 105))
        if enemy == ENEMIES[1]:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        if enemy == ENEMIES[0]:
            plt.ylabel("Population fitness")
        plt.xlabel("Generation")
        plt.title("Enemy {}".format(enemy))

    plt.show()

        
    





    

    
