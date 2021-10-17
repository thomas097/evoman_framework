
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
import sys, os
sys.path.insert(0, 'evoman')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pickle
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from evaluation import eval_population, eval_individual
from demo_controller import player_controller
from environment import Environment

if __name__ == "__main__":

    RUNS = 10
    ENEMIES = [[1, 5, 7], [2, 3, 4]]
    FITNESSES = [1, 2]
    FILENAME = "solutions/best_run-{}_enemies-{}_fitness-{}.txt"

    for enemies in ENEMIES:
        for fitness_type in FITNESSES:

            runtimes = []
            for run in range(RUNS):

                # Load weights from file.
                filename = FILENAME.format(run, enemies, fitness_type)
                weights = np.loadtxt(filename)

                # Play game against all available enemies.
                for enemy in range(1, 9):

                    env = Environment(experiment_name=None,
                                          enemies=[enemy],
                                          player_controller=player_controller(_n_hidden=10),
                                          contacthurt='player',
                                          speed='fastest',
                                          logs="off",
                                          randomini='no',
                                          level=2)

                    # Average performance over 5 runs
                    _, _, _, duration = env.play(pcont=weights)
                    runtimes.append(duration)

            avg_runtime = np.mean(runtimes)
            print("enemies =", enemies, "Fitness =", fitness_type, "Average run time", avg_runtime)


    # Stats test
    for enemies in ENEMIES:

        runtimes_fitness1 = []
        runtimes_fitness2 = []

        # Play game against all available enemies.
        for enemy in range(1, 9):

            runtime_fitness1 = []
            runtime_fitness2 = []
            for run in range(RUNS):

                # Load weights from file.
                filename1 = FILENAME.format(run, enemies, 1)
                weights1 = np.loadtxt(filename1)

                filename2 = FILENAME.format(run, enemies, 2)
                weights2 = np.loadtxt(filename2)

                env = Environment(experiment_name=None,
                                      enemies=[enemy],
                                      player_controller=player_controller(_n_hidden=10),
                                      contacthurt='player',
                                      speed='fastest',
                                      logs="off",
                                      randomini='no',
                                      level=2)

                # Average performance over 5 runs
                _, _, _, duration1 = env.play(pcont=weights1)
                runtime_fitness1.append(duration1)

                _, _, _, duration2 = env.play(pcont=weights2)
                runtime_fitness2.append(duration2)

            runtimes_fitness1.append(np.mean(runtime_fitness1) - 20)
            runtimes_fitness2.append(np.mean(runtime_fitness2))

        t, p = ttest_rel(runtimes_fitness2, runtimes_fitness1)
        print("enemies =", enemies, "t =", t, "p =", p)  
        
         

    


