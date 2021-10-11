import numpy as np

class Logger:
    """ Convenience class to log the mean and max fitness of
        genotypes in the population over the course of a run.
    """

    def __init__(self, run, enemies, fitness):
        # Set up file with header.
        self.stats_fname = "stats_run-{}_enemies-{}_fitness-{}".format(run, enemies, fitness)
        with open(self.stats_fname + ".csv", "w") as f:
            f.write("mean,max\n")

    def log(self, pop_fitnesses):
        # Print population statistics.
        mean = np.mean(pop_fitnesses)
        _max = np.max(pop_fitnesses)
        print("Stats: MEAN={} MAX={}".format(mean, _max))

        # Write stats to file.
        with open(self.stats_fname + ".csv", "a") as f:
            f.write("{},{}\n".format(mean, _max))

    def save_best(self, pop, pop_fitnesses):
        # Write best solution to file.
        solution = pop[np.argmax(pop_fitnesses)]
        solution = solution[:265] # Strip sigma if exists
        solution_fname = self.stats_fname.replace("stats", "best")
        np.savetxt(solution_fname + ".txt", solution)
