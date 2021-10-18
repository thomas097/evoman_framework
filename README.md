This folder contains all code used to generate the results of Assignment 2 (Task II: generalist agent) for the 
Evolutionary Computing 2021 course at the Vrije Universiteit (VU) Amsterdam.

For this assignment we implemented evolutionary algorithms with different fitness functions, differing in their
inclusion of a time penalty. This README will hopefully provide some insight regarding our implementation of the 
algorithms and will explain how to run our files.

# Dependencies
Other than the standard libraries we include:
* tqdm
* scipy

# Overview of files
The following files and directories are included in this repository:

## Directories:
* \evoman: 				Implementation of the Evoman environment (not modified in any way).
* \fitness_A-group_B: 			Folder containing all solution files and statistics files for fitness
					function A (1 or 2) and group B (1 or 2).

Note: In order to run generalist_boxplot.py, generalist_fitness_plots.py, generalist_outcome_table.py, and 
generalist_time_table.py, the solution txt files must be placed in a folder \solutions in the root directory 
\19. The same holds for the fitness plots, for which the stats files need to be placed in a folder \stats. 
For sake of file size, this was not done.

## Code files:
* generalist_boxplot.py			Runs the solutions in the \solutions folder 5 times and creates a box-
					plot for each algorithm and enemy showing their gain scores over enemies.	
* generalist_fitness_plots.py		Plots the statistics (mean and stdev of max/mean fitness during a run) 
					stored in the csv files in the \stats folder.
* generalist_NN_training.py		Runs the evolutionary algorithm for x amount of runs and y generations 
					with user specified enemy group and fitness function.
* generalist_outcome_table.py		Plots the player energy and enemy energy after a match with the best 
					controller for each enemy in the game. 
* generalist_time_table.py		Evaluates the average run time of each group of runs per fitness 
					function and enemy group.
* demo_controller.py			Unaltered demo controller file provided by the Evoman framework.


To facilitate easy collaboration, the algorithm itself was split over several files:
* logger.py				Keeps track of statistics during a run and saves final best solution.
* mutation.py				Implements mutation operators
* parent_selection(2).py		Implements parent selection and fitness sharng
* recombination.py			Implements crossover operators
* survivor selection.py			Implements survivor selection operators
* evaluation.py				Contains all code used to evaluate individuals.

The final best solution is provided as 19.txt.

# Usage

## Running the EAs
To run the algorithms with the standard settings used by us in the report (using enemies (1,5,7)
and fitness function 1), use the following command:

Windows:

$ py -3 generalist_NN_training.py --runs=10 --generations=75 --enemies=1,5,7 --fitness=1

Ubuntu:

$ python3 generalist_NN_training.py --runs=10 --generations=75 --enemies=1,5,7 --fitness=1

To select the second enemy group (fitness function), use --enemies=2,3,4 (--fitness=2) instead.


## Visualizing the EAs
To visualize the progress plots and boxplots, add the txt solutions to \solutions and statistics csv 
files to \stats. Then simply run the following command:

Windows:

$ py -3 generalist_boxplot.py
$ py -3 generalist_fitness_plots.py

Ubuntu:

$ python3 generalist_boxplot.py
$ python3 generalist_fitness_plots.py

