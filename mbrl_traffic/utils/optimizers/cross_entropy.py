"""Script containing the cross-entropy method optimizer."""
from mbrl_traffic.utils.optimizers.base import Optimizer
import numpy as np

class CrossEntropyMethod(Optimizer):  # TODO
    """Cross-entropy method optimizer object. Adopted from
    https://github.com/jerrylin1121/cross_entropy_method"""

    def __init__(self,
                 param_low,
                 param_high,
                 fitness_fn,
                 dim,
                 verbose=2,
                 samples=5,
                 ne=10,
                 argmin=True,
                 init_scale=1,
                 samplemethod='Uniform'):
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
        self.func = fitness_fn  # target function
        self.d = dim  # dimension of function input X: len(x0)
        self.N = samples  # sample N examples each iteration
        self.Ne = ne  # using better Ne examples to update mu and sigma
        self.reverse = not argmin  # try to maximum or minimum the target function (not argmin finds the minimum)
        self.v_min = param_low  # the value minimum
        self.v_max = param_high  # the value maximum
        self.init_coef = init_scale  # sigma initial value

        assert samplemethod == 'Gaussian' or samplemethod == 'Uniform'
        self.sampleMethod = samplemethod  # which sample method gaussian or uniform, default to gaussian

    def solve(self, num_steps=1000, termination_fn=None):
        """See parent class."""
        # specify the number of steps this will run
        self.maxits = num_steps

        sol = self.eval()

        return sol, self.func(sol)

    def eval(self):
        """evaluation and return the solution"""
        if self.sampleMethod == 'Gaussian':
            return self.eval_gaussian()
        elif self.sampleMethod == 'Uniform':
            return self.eval_uniform()

    def eval_uniform(self):

        t, _min, _max = self.__init_uniform_params()

        # random sample all dimension each time
        while t < self.maxits:
            # sample N data and sort
            x = self.__uniform_sample_data(_min, _max)
            s = self.__function_reward(x)
            s = self.__sort_sample(s)
            x = np.array([s[i][0] for i in range(np.shape(s)[0])])
            _min, _max = self.__update_uniform_params(x)
            t += 1

        return (_min + _max) / 2.

    def eval_gaussian(self):
        # initial parameters
        t, mu, sigma = self.__init_gaussian_params()

        # random sample all dimension each time
        while t < self.maxits:
            # sample N data and sort
            x = self.__gaussian_sample_data(mu, sigma)
            s = self.__function_reward(x)
            s = self.__sort_sample(s)
            x = np.array([s[i][0] for i in range(np.shape(s)[0])])

            # update parameters
            mu, sigma = self.__update_gaussian_params(x)
            t += 1

        return mu

    def __init_gaussian_params(self):
        """initial parameters t, mu, sigma"""
        t = 0
        mu = np.zeros(self.d)
        sigma = np.ones(self.d) * self.init_coef
        return t, mu, sigma

    def __update_gaussian_params(self, x):
        """update parameters mu, sigma"""
        mu = x[0:self.Ne, :].mean(axis=0)
        sigma = x[0:self.Ne, :].std(axis=0)
        return mu, sigma

    def __gaussian_sample_data(self, mu, sigma):
        """sample N examples"""
        sample_matrix = np.zeros((self.N, self.d))
        for j in range(self.d):
            sample_matrix[:, j] = np.random.normal(loc=mu[j], scale=sigma[j] + 1e-17, size=(self.N,))
            if self.v_min is not None and self.v_max is not None:
                sample_matrix[:, j] = np.clip(sample_matrix[:, j], self.v_min[j], self.v_max[j])
        return sample_matrix

    def __init_uniform_params(self):
        """initial parameters t, mu, sigma"""
        t = 0
        _min = self.v_min if self.v_min else -np.ones(self.d)
        _max = self.v_max if self.v_max else np.ones(self.d)
        return t, _min, _max

    def __update_uniform_params(self, x):
        """update parameters mu, sigma"""
        _min = np.amin(x[0:self.Ne, :], axis=0)
        _max = np.amax(x[0:self.Ne, :], axis=0)
        return _min, _max

    def __uniform_sample_data(self, _min, _max):
        """sample N examples"""
        sample_matrix = np.zeros((self.N, self.d))
        for j in range(self.d):
            sample_matrix[:, j] = np.random.uniform(low=_min[j], high=_max[j], size=(self.N,))
        return sample_matrix

    def __function_reward(self, x):

        # self.func (fitness_func) should support matrix  operations for faster computations
        value = []
        for i in np.arange(len(x)):
            # evaluate the value of each guess (x)
            val2 = self.func(x[i])
            value = np.append(value, val2)

        return zip(x, value)

    def __sort_sample(self, s):
        """sort data by function return"""
        s = sorted(s, key=lambda x: x[1], reverse=self.reverse)

        return s


def sqr(x):
    # solution x = 2 and y(x) = 3
    return (x-2) ** 2 + 3


def loss_func(x_pred, guess=0):
    if len(x_pred) > 1:
        x_pred = x_pred[0]
    return np.sqrt((sqr(x_pred) - sqr(guess))**2)


bds_low = [0, 5]
bds_high = [5, 7]
op = CrossEntropyMethod(bds_low, bds_high, loss_func, dim=2)
solution = op.solve(num_steps=50000)

print(solution)
