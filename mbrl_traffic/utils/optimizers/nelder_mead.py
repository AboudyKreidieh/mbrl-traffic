"""Script containing the Nelder Mead optimizer."""
from mbrl_traffic.utils.optimizers.base import Optimizer
import scipy.optimize as optimize


class NelderMead(Optimizer):
    """Nelder Mead optimizer object.

    Attributes
    ----------
    x0 : array_like
        the initial guess for the solver
    bnds : (array_like, array_like)
        lower and upper bounds for each other parameters
    options : dict
        additional solver options that can be passed in
    """

    def __init__(self,
                 param_low,
                 param_high,
                 fitness_fn,
                 x0,
                 verbose=2,
                 disp=True,
                 initial_simplex=None,
                 xatol=1e-8,
                 fatol=1000,
                 adaptive=False):
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
        disp : bool
            Set to True to print convergence messages.
        initial_simplex : array_like of shape (N+1, N)
            Initial simplex. If given, overrides x0. initial_simplex[j,:] 
            should contain the coordinates of the j-th vertex of the N+1
            vertices in the simplex, where N is the dimension.
        xatol : float
            Absolute error in xopt between iterations that is acceptable for
            convergence.
        fatol : float
            Absolute error in func(xopt) between iterations that is acceptable
            for convergence.
        adaptive : bool
            Adapt algorithm parameters to dimensionality of problem. Useful
            for high-dimensional minimization
        """
        super(NelderMead, self).__init__(
            param_low=param_low,
            param_high=param_high,
            fitness_fn=fitness_fn,
            verbose=verbose
        )

        self.x0 = x0
        self.bnds = (param_low, param_high)
        self.options = {
            'disp': disp,
            'initial_simplex': initial_simplex,
            'xatol': xatol,
            'fatol': fatol,
            'adaptive': adaptive
        }

    def solve(self, num_steps=1000, termination_fn=None):
        """See parent class."""
        # specify the number of steps this will run
        self.options['maxiter'] = num_steps

        sol = optimize.minimize(
            self.fitness_fn,
            self.x0,
            method="Nelder-Mead",
            bounds=self.bnds,
            callback=None,
            options=self.options
        )

        return sol.x
