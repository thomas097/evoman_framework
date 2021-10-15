
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
import neat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from evaluation import eval_population, eval_individual
from demo_controller import player_controller
from environment import Environment

if __name__ == "__main__":

    REPEATS = 5
    FILENAME = "best_run-6_enemies-[2, 3, 4]_fitness-1.txt"

    # Load weights from file.
    weights = np.loadtxt(FILENAME)

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
        player_energy = 0
        enemy_energy = 0
        for i in range(REPEATS):
            res = env.play(pcont=weights)
            player_energy += res[1] / REPEATS
            enemy_energy += res[2] / REPEATS
            

        # Run game with genome weights (exclude sigma).
        print("enemy:", enemy,
              "\t", round(player_energy, 3),
              "\t", round(enemy_energy, 3))
         

    


