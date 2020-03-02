import numpy as np
import heapq

from mbrl_traffic.utils.optimizers.base import Optimizer


class GeneticAlgorithm(Optimizer):
    """Genetic algorithm optimizer object.

    This algorithm is heavily adopted from the following link:
    https://towardsdatascience.com/continuous-genetic-algorithm-from-scratch-with-python-ff29deedd099

    Attributes
    ----------
    population_size : int
        the number of individual parameter sets used during the optimization
        procedure
    selection_method : str
        TODO
    pairing_method : str
        TODO
    mating_method : str
        TODO
    mutation_method : str
        TODO
    mutation_prob : float
        TODO
    mutation_std : float
        TODO
    """

    def __init__(self,
                 param_low,
                 param_high,
                 fitness_fn,
                 population_size=100,
                 selection_method="fittest",
                 pairing_method="fittest",
                 mating_method="one-point",
                 mutation_method="gauss",
                 mutation_prob=0.01,
                 mutation_std=0.001):
        """Instantiate the base optimizer.

        Parameters
        ----------
        param_low : array_like
            the minimum value for each parameter
        param_high : array_like
            the maximum value for each parameter
        fitness_fn : function
            the fitness function. Takes as input a set of values for each
            parameter
        population_size : int
            the number of individual parameter sets used during the
            optimization procedure
        selection_method : str
            TODO
        pairing_method : str
            TODO
        mating_method : str
            TODO
        mutation_method : str
            TODO
        mutation_prob : float
            TODO
        mutation_std : float
            TODO
        """
        super(GeneticAlgorithm, self).__init__(
            param_low=param_low,
            param_high=param_high,
            fitness_fn=fitness_fn
        )
        self.population_size = population_size
        self.selection_method = selection_method
        self.pairing_method = pairing_method
        self.mating_method = mating_method
        self.mutation_method = mutation_method
        self.mutation_prob = mutation_prob
        self.mutation_std = mutation_std

    def solve(self, num_steps=1, termination_fn=None):
        """See parent class."""
        fitness = np.zeros(self.population_size)

        # =================================================================== #
        # Step 1. Initialize the population.                                  #
        # =================================================================== #
        population = np.random.uniform(
            low=self.param_low,
            high=self.param_high,
            size=self.population_size
        )

        for _ in range(num_steps):
            # =============================================================== #
            # Step 2. Calculate the fitness of the current samples.           #
            # =============================================================== #
            fitness = np.array(self.fitness_fn(population[:, i])
                               for i in range(self.population_size))
            total_fitness = sum(fitness)
            cum_sum = np.cumsum(fitness) / total_fitness

            # =============================================================== #
            # Step 3. Test for the termination condition.                     #
            # =============================================================== #
            if termination_fn is not None and termination_fn(fitness):
                break

            # =============================================================== #
            # Step 4. Perform the selection procedure.                        #
            # =============================================================== #
            selection = np.empty((len(self.param_low), self.population_size))
            selection_size = round(self.population_size / 2)

            if self.selection_method == "fittest":
                # Choose the parameters with the highest fitness values.
                idx = heapq.nlargest(selection_size,
                                     range(len(fitness)), fitness.take)
                selection[:, len(idx)] = population[:, idx]

            elif self.selection_method == "roulette":
                # Choose the parameters randomly weighted by the fitness of the
                # individuals in the population.
                idx = np.random.uniform(0, 1, selection_size)
                idx = np.array(np.searchsorted(idx[i]) - 1 for i in idx.size)
                selection[:, len(idx)] = population[:, idx]

            elif self.selection_method == "random":
                # Choose the parameters random uniformly.
                population_size = self.population_size
                idx = np.random.randint(0, population_size, selection_size)
                selection[:, len(idx)] = population[:, idx]

            # =============================================================== #
            # Step 5. Perform the pairing procedure.                          #
            # =============================================================== #
            if self.pairing_method == "fittest":
                pass

            elif self.pairing_method == "random":
                pass

            elif self.pairing_method == "weighted-random":
                pass

            # =============================================================== #
            # Step 6. Perform the mating procedure.                           #
            # =============================================================== #
            if self.mating_method == "one-point":
                pass

            elif self.mating_method == "two-points":
                pass

            # =============================================================== #
            # Step 6. Perform the mutation procedure.                         #
            # =============================================================== #
            if self.mutation_method == "gauss":
                pass

            elif self.mutation_method == "reset":
                pass

            # =============================================================== #
            # Step 7. Create the next generation.                             #
            # =============================================================== #
            pass

        # Return the individual with the best fitness.
        return population[:, np.argmax(fitness)]
