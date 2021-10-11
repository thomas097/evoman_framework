import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from tqdm import tqdm
import numpy as np


# Evaluates each genome in the population.
def eval_population(pop, fitness_func, enemies, num_trials=1):
    pop_fitnesses = []
    for ind in tqdm(pop):

        # Average fitness over x trials for y enemies.
        ind_fitness = []
        for enemy in enemies:
            for _ in range(num_trials):
                ind_fitness.append( eval_individual(ind, fitness_func, enemy)[0] )
        pop_fitnesses.append(np.mean(ind_fitness))

    return np.array(pop_fitnesses)



def eval_individual(genome, fitness_func, enemy):
    # Set up environment for enemy.
    env = Environment(experiment_name=None,
                      enemies=[enemy],
                      player_controller=player_controller(_n_hidden=10),
                      contacthurt='player',
                      speed='fastest',
                      logs="off",
                      randomini='yes',
                      level=2)

    # Run game with genome weights (exclude sigma).
    weights = genome[:265]
    game_fitness, player_energy, enemy_energy, time = env.play(pcont=weights)
    gain = player_energy - enemy_energy

    # Original fitness function
    if fitness_func == 1:
        fitness = fitness_1(player_energy,enemy_energy)

    # Alternative fitness function
    elif fitness_func == 2:
        fitness = fitness_2(player_energy,enemy_energy, time)

    return fitness, gain


def fitness_1(player_energy, enemy_energy):
    """
    remove the consideration of time from the original fitness function
    0.9(100-e) + 0.1p
    """
    return 0.9*(100-enemy_energy) + 0.1*(player_energy)


def fitness_2(player_energy, enemy_energy, time):
    """
    original fitness function
    0.9(100-e) + 0.1p -log t
    """
    return 0.9*(100-enemy_energy) + 0.1*(player_energy) - np.log(time)
