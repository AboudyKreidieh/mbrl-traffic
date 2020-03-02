from mbrl_traffic.utils.optimizers.base import Optimizer


class CrossEntropyMethod(Optimizer):
    """

    Attributes
    ----------
    TODO
        TODO
    """

    def __init__(self,
                 param_low,
                 param_high,
                 fitness_fn):
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
        """
        super(CrossEntropyMethod, self).__init__(
            param_low=param_low,
            param_high=param_high,
            fitness_fn=fitness_fn
        )

    def solve(self, num_steps=1000, termination_fn=None):
        """See parent class."""
        pass
