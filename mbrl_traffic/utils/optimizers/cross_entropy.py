"""Script containing the cross-entropy method optimizer."""
import numpy as np

from mbrl_traffic.utils.optimizers.base import Optimizer


class CrossEntropyMethod(Optimizer):
    """Cross-entropy method optimizer object.

    Adopted from: https://github.com/jerrylin1121/cross_entropy_method

    Attributes
    ----------
    d : int
        dimension of function input X: len(x0)
    N : int
        sample N examples each iteration
    Ne : int
        using better Ne examples to update mu and sigma
    reverse : bool
        try to maximum or minimum the target function (reverse finds the
        minimum)
    init_scale : float
        sigma initial value
    sample_method : str
        which sample method gaussian or uniform, default to gaussian
    """

    def __init__(self,
                 param_low,
                 param_high,
                 fitness_fn,
                 verbose=2,
                 samples=5,
                 ne=10,
                 argmin=True,
                 init_scale=1,
                 sample_method='uniform'):
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
        samples : int
            sample N examples each iteration
        ne : int
            using better Ne examples to update mu and sigma
        argmin : bool
            try to maximum or minimum the target function (not argmin finds the
            minimum)
        init_scale : float
            sigma initial value
        sample_method : str
            which sample method gaussian or uniform, default to gaussian
        """
        super(CrossEntropyMethod, self).__init__(
            param_low=param_low,
            param_high=param_high,
            fitness_fn=fitness_fn,
            verbose=verbose
        )
        self.d = len(param_low)
        self.N = samples
        self.Ne = ne
        self.reverse = not argmin
        self.init_coef = init_scale

        assert sample_method == 'gaussian' or sample_method == 'uniform'
        self.sample_method = sample_method

    def solve(self, num_steps=1000, termination_fn=None):
        """See parent class."""
        if self.sample_method == 'gaussian':
            sol = self.eval_gaussian(num_steps)
        elif self.sample_method == 'uniform':
            sol = self.eval_uniform(num_steps)
        else:
            sol = None

        return sol, self.fitness_fn(sol)

    def eval_uniform(self, num_steps):
        """Perform the uniform training operation."""
        t, _min, _max = self._init_uniform_params()

        # random sample all dimension each time
        while t < num_steps:
            # sample N data and sort
            x = self._uniform_sample_data(_min, _max)
            s = self._function_reward(x)
            s = self._sort_sample(s)
            x = np.array([s[i][0] for i in range(np.shape(s)[0])])
            _min, _max = self._update_uniform_params(x)
            t += 1

        return (_min + _max) / 2.

    def eval_gaussian(self, num_steps):
        """Perform the gaussian training operation."""
        # initial parameters
        t, mu, sigma = self._init_gaussian_params()

        # random sample all dimension each time
        while t < num_steps:
            # sample N data and sort
            x = self._gaussian_sample_data(mu, sigma)
            s = self._function_reward(x)
            s = self._sort_sample(s)
            x = np.array([s[i][0] for i in range(np.shape(s)[0])])

            # update parameters
            mu, sigma = self._update_gaussian_params(x)
            t += 1

        return mu

    def _init_gaussian_params(self):
        """Initialize the t, mu, and sigma parameters."""
        t = 0
        mu = np.zeros(self.d)
        sigma = np.ones(self.d) * self.init_coef
        return t, mu, sigma

    def _update_gaussian_params(self, x):
        """Update the mu and sigma parameters."""
        mu = x[0:self.Ne, :].mean(axis=0)
        sigma = x[0:self.Ne, :].std(axis=0)
        return mu, sigma

    def _gaussian_sample_data(self, mu, sigma):
        """Sample N examples."""
        sample_matrix = np.zeros((self.N, self.d))
        for j in range(self.d):
            sample_matrix[:, j] = np.random.normal(
                loc=mu[j], scale=sigma[j] + 1e-17, size=(self.N,))
            if self.param_low is not None and self.param_high is not None:
                sample_matrix[:, j] = np.clip(
                    sample_matrix[:, j], self.param_low[j], self.param_high[j])
        return sample_matrix

    def _init_uniform_params(self):
        """Initialize the t, mu, and sigma parameters."""
        t = 0
        _min = self.param_low if self.param_low else -np.ones(self.d)
        _max = self.param_high if self.param_high else np.ones(self.d)
        return t, _min, _max

    def _update_uniform_params(self, x):
        """Update the mu and sigma parameters."""
        _min = np.amin(x[0:self.Ne, :], axis=0)
        _max = np.amax(x[0:self.Ne, :], axis=0)
        return _min, _max

    def _uniform_sample_data(self, _min, _max):
        """Sample N examples."""
        sample_matrix = np.zeros((self.N, self.d))
        for j in range(self.d):
            sample_matrix[:, j] = np.random.uniform(
                low=_min[j], high=_max[j], size=(self.N,))
        return sample_matrix

    def _function_reward(self, x):
        """Compute the fitness of each sample and return alongside sample."""
        value = []
        for i in np.arange(len(x)):
            # evaluate the value of each guess (x)
            val2 = self.fitness_fn(x[i])
            value = np.append(value, val2)

        return zip(x, value)

    def _sort_sample(self, s):
        """Sort data by function return."""
        return sorted(s, key=lambda x: x[1], reverse=self.reverse)


def sqr(x):
    # solution x = 2 and y(x) = 3
    return (x-2) ** 2 + 3


def loss_func(x_pred, guess=0):
    if len(x_pred) > 1:
        x_pred = x_pred[0]
    return np.sqrt((sqr(x_pred) - sqr(guess))**2)


bds_low = [0, 5]
bds_high = [5, 7]
op = CrossEntropyMethod(bds_low, bds_high, loss_func)
solution = op.solve(num_steps=50000)

print(solution)
