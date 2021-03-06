"""Script containing the cross-entropy method optimizer."""
from mbrl_traffic.utils.optimizers.base import Optimizer


class CrossEntropyMethod(Optimizer):
    """Cross-entropy method optimizer object."""

    def __init__(self,
                 param_low,
                 param_high,
                 fitness_fn,
                 verbose=2):
        """Instantiate the optimizer.

        Parameters
        ----------
        param_low : array_like
            the minimum value for each parameter
        param_high : array_like
            the maximum value for each parameter
        fitness_fn : function
            the fitness function. Takes as input a set of values for each
            parameter
        verbose : int
            the verbosity flag
        """
        super(CrossEntropyMethod, self).__init__(
            param_low=param_low,
            param_high=param_high,
            fitness_fn=fitness_fn,
            verbose=verbose
        )

    def solve(self, num_steps=1000, termination_fn=None):
        """See parent class."""
        pass
