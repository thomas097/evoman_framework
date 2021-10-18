"""
    Filename:    generalist_fitness_plots.py
    Author(s):   Thomas Bellucci, Svetlana Codrean
    Assignment:  Task 2 - Evolutionary Computing
    Description: Creates the fitness curves for the final report
                 showing the mean and max fitness for the population
                 as a function of the generation. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

if __name__ == "__main__":

    ENEMIES = [[1, 5, 7],
               [2, 3, 4]]
    FITNESS_TYPES = [1, 2]
    GENS = 75
    RUNS = 10

    for k, enemies in enumerate(ENEMIES):

        mean_EA1 = []
        mean_EA2 = []
        max_EA1 = []
        max_EA2 = []

        # Load in mean and max stats for each run.
        for run in range(RUNS):
            for fitness_type in FITNESS_TYPES:

                filename = "stats/stats_run-{}_enemies-{}_fitness-{}.csv".format(run, enemies, fitness_type)
                df = pd.read_csv(filename)

                if fitness_type == 1:
                    mean_EA1.append(df["mean"])
                    max_EA1.append(df["max"])
                    
                elif fitness_type == 2:
                    mean_EA2.append(df["mean"])
                    max_EA2.append(df["max"])

        # Compute mean and stdev over runs of mean and max population stats.
        x = np.arange(GENS + 1)
        mean_mean_EA1 = np.mean(mean_EA1, axis=0)
        mean_mean_EA2 = np.mean(mean_EA2, axis=0)
        mean_max_EA1 = np.mean(max_EA1, axis=0)
        mean_max_EA2 = np.mean(max_EA2, axis=0)

        std_mean_EA1 = np.std(mean_EA1, axis=0)
        std_mean_EA2 = np.std(mean_EA2, axis=0)
        std_max_EA1 = np.std(max_EA1, axis=0)
        std_max_EA2 = np.std(max_EA2, axis=0)

        # Plot mean and stdev of mean population fitness.
        plt.subplot(1, 2, k+1)
        plt.plot(x, mean_mean_EA1, label="EA1 (mean)", c="C0")
        plt.fill_between(x, mean_mean_EA1 - std_mean_EA1, mean_mean_EA1 + std_mean_EA1, alpha=0.4, fc="C0")
        
        plt.plot(x, mean_mean_EA2, label="EA2 (mean)", c="C1")
        plt.fill_between(x, mean_mean_EA2 - std_mean_EA2, mean_mean_EA2 + std_mean_EA2, alpha=0.4, fc="C1")

        # Plot mean and stdev of max population fitness.
        plt.subplot(1, 2, k+1)
        plt.plot(x, mean_max_EA1, label="EA1 (max)", c="C0", linestyle="--")
        plt.fill_between(x, mean_max_EA1 - std_max_EA1, mean_max_EA1 + std_max_EA1, alpha=0.4, fc="C0")
        
        plt.plot(x, mean_max_EA2, label="EA2 (max)", c="C1", linestyle="--")
        plt.fill_between(x, mean_max_EA2 - std_max_EA2, mean_max_EA2 + std_max_EA2, alpha=0.4, fc="C1")

        if k == 1:
            plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.13), ncol=2)
        if k == 0:
            plt.ylabel("Population fitness")
        plt.xlabel("Generation")
        plt.title("Enemy Group {}".format(enemies))

    plt.show()

    





    

    
