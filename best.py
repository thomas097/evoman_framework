best_guy = r"E:\Libraries\Repos\evo_comp2\evoman_framework\fitness_1-group_1\best-ind-enemies-1,5,7_fitness-1\best_run-6_enemies-[1, 5, 7]_fitness-1.txt"
best_guy2 = r"E:\Libraries\Repos\evo_comp2\evoman_framework\fitness_1-group_2\best-ind-enemies-2,3,4_fitness-1\best_run-0_enemies-[2, 3, 4]_fitness-1.txt"
best_guy3 = r"E:\Libraries\Repos\evo_comp2\evoman_framework\fitness_2-group_1\best-ind-enemies-1,5,7_fitness-2\best_run-0_enemies-[1, 5, 7]_fitness-2.txt"
best_guy4 = r"E:\Libraries\Repos\evo_comp2\evoman_framework\fitness_2-group_2\best-ind-enemies-2,3,4_fitness-2\best_run-0_enemies-[2, 3, 4]_fitness-2.txt"
from evaluation import eval_individual
import numpy as np, pandas as pd
genome = np.loadtxt(best_guy)
genome2 = np.loadtxt(best_guy2)
genome3 = np.loadtxt(best_guy3)
genome4 = np.loadtxt(best_guy4)
enemies = [1,5,7]
enemies2 = [2,3,4]
gains = pd.DataFrame({f"run_{run}" : [eval_individual(genome, 1, enemy)[1] for enemy in enemies] for run in range(1,6)},
index=enemies)

gains2 = pd.DataFrame({f"run_{run}" : [eval_individual(genome2, 1, enemy)[1] for enemy in enemies2] for run in range(1,6)},
index=enemies2)

gains3 = pd.DataFrame({f"run_{run}" : [eval_individual(genome3, 2, enemy)[1] for enemy in enemies] for run in range(1,6)},
index=enemies)

gains4 = pd.DataFrame({f"run_{run}" : [eval_individual(genome4, 2, enemy)[1] for enemy in enemies2] for run in range(1,6)},
index=enemies2)

result = pd.DataFrame({"group1_fitness1":gains.mean().values, "group2_fitness1":gains2.mean().values, "group1_fitness2":gains3.mean().values, "group2_fitness2":gains4.mean().values})
print(result)

plot = result.boxplot()
fig = plot.get_figure()
fig.savefig("boxplot.png")
