"""Base optimizer object for macroscopic models."""


class Optimizer(object):
    """Base optimizer object.

    This object is used to train the parameters of a given model to maximize
    some fitness function. Examples of optimizers implemented can be found
    within the same folder as this script.

    Attributes
    ----------
    param_low : array_like
        the minimum value for each parameter
    param_high : array_like
        the maximum value for each parameter
    fitness_fn : function
        the fitness function. Takes as input a set of values for each parameter
    """

    def __init__(self, param_low, param_high, fitness_fn):
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
        """
        self.param_low = param_low
        self.param_high = param_high
        self.fitness_fn = fitness_fn

    def solve(self, num_steps=1, termination_fn=None):
        """Perform the training procedure.

        Parameters
        ----------
        num_steps : int
            maximum number of training iterations
        termination_fn : function or None
            an early termination function, for example to check if the fitness
            has reached a certain threshold. Takes as input the fitness of the
            currently stored parameters. If set to None, this term is ignored
            in the training procedure.

        Returns
        -------
        array_like
            the optimal parameters
        float
            the fitness associated with the provided parameters
        """
        raise NotImplementedError
