
"""
    Filename:    generalist_time_table.py
    Author(s):   Thomas Bellucci, Svetlana Codrean
    Assignment:  Task 2 - Evolutionary Computing
    Description: Creates the time table with average run times
                 for the algorithm variants.
"""
import sys, os
sys.path.insert(0, 'evoman')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pickle
import numpy as np

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
        
         

    


