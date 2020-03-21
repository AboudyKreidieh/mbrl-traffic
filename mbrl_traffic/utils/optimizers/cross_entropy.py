"""Script containing the cross-entropy method optimizer."""
from mbrl_traffic.utils.optimizers.base import Optimizer
import numpy as np
import random

class CrossEntropyMethod(Optimizer):  # TODO
    """Cross-entropy method optimizer object."""

    def __init__(self,
                 param_low,
                 param_high,
                 fitness_fn,
                 x0,
                 verbose=2,
                 samples=5,
                 ne=10,
                 argmin=True,
                 init_scale=1,
                 samplemethod='Gaussian'):
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
        self.d = len(x0)  # dimension of function input X: len(x0)
        self.N = samples  # sample N examples each iteration
        self.Ne = ne  # using better Ne examples to update mu and sigma
        self.reverse = not argmin  # try to maximum or minimum the target function (not argmin finds the minimum)
        self.v_min = param_low  # the value minimum
        self.v_max = param_high  # the value maximum
        self.init_coef = init_scale  # sigma initial value
        self.x0 = x0  # np.array like

        assert samplemethod == 'Gaussian' or samplemethod == 'Uniform'
        self.sampleMethod = samplemethod  # which sample method gaussian or uniform, default to gaussian

    def solve(self, num_steps=1000, termination_fn=None):
        """See parent class."""
        # specify the number of steps this will run
        self.maxits = num_steps

        sol = self.eval(self.x0)

        return sol, self.func(sol)

    def eval(self, instr):
        """evalution and return the solution"""
        if self.sampleMethod == 'Gaussian':
            return self.evalGaussian(instr)
        elif self.sampleMethod == 'Uniform':
            return self.evalUniform(instr)

    def evalUniform(self, instr):

        t, _min, _max = self.__initUniformParams()

        # random sample all dimension each time
        while t < self.maxits:
            # sample N data and sort
            x = self.__uniformSampleData(_min, _max)
            s = self.__functionReward(instr, x)
            s = self.__sortSample(s)
            x = np.array([s[i][0] for i in range(np.shape(s)[0])])
            l = np.array([s[i][1] for i in range(np.shape(s)[0])])

            _min, _max = self.__updateUniformParams(x)
            t += 1

        return (_min + _max) / 2.

    def evalGaussian(self, instr):
        # initial parameters
        t, mu, sigma = self.__initGaussianParams()

        # random sample all dimension each time
        while t < self.maxits:
            # sample N data and sort
            x = self.__gaussianSampleData(mu, sigma)
            s = self.__functionReward(instr, x)
            s = self.__sortSample(s)
            x = np.array([s[i][0] for i in range(np.shape(s)[0])])

            # update parameters
            mu, sigma = self.__updateGaussianParams(x)
            t += 1

        return mu

    def __initGaussianParams(self):
        """initial parameters t, mu, sigma"""
        t = 0
        mu = np.zeros(self.d)
        sigma = np.ones(self.d) * self.init_coef
        return t, mu, sigma

    def __updateGaussianParams(self, x):
        """update parameters mu, sigma"""
        mu = x[0:self.Ne, :].mean(axis=0)
        sigma = x[0:self.Ne, :].std(axis=0)
        return mu, sigma

    def __gaussianSampleData(self, mu, sigma):
        """sample N examples"""
        sample_matrix = np.zeros((self.N, self.d))
        for j in range(self.d):
            sample_matrix[:, j] = np.random.normal(loc=mu[j], scale=sigma[j] + 1e-17, size=(self.N,))
            if self.v_min is not None and self.v_max is not None:
                sample_matrix[:, j] = np.clip(sample_matrix[:, j], self.v_min[j], self.v_max[j])
        return sample_matrix

    def __initUniformParams(self):
        """initial parameters t, mu, sigma"""
        t = 0
        _min = self.v_min if self.v_min else -np.ones(self.d)
        _max = self.v_max if self.v_max else np.ones(self.d)
        return t, _min, _max

    def __updateUniformParams(self, x):
        """update parameters mu, sigma"""
        _min = np.amin(x[0:self.Ne, :], axis=0)
        _max = np.amax(x[0:self.Ne, :], axis=0)
        return _min, _max

    def __uniformSampleData(self, _min, _max):
        """sample N examples"""
        sample_matrix = np.zeros((self.N, self.d))
        for j in range(self.d):
            sample_matrix[:, j] = np.random.uniform(low=_min[j], high=_max[j], size=(self.N,))
        return sample_matrix

    def __functionReward(self, instr, x):
        bi = np.reshape(instr, [1, -1])
        bi = np.repeat(bi, self.N, axis=0)

        # self.func (fitness_func) should support matrix  operations for faster computations
        value = []
        for i in np.arange(len(x)):
            # evaluate the value of each guess (x)
            val2 = self.func(x[i])
            value = np.append(value, val2)

        return zip(x, value)

    def __sortSample(self, s):
        """sort data by function return"""
        s = sorted(s, key=lambda x: x[1], reverse=self.reverse)

        return s
