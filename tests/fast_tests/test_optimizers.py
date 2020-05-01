"""Tests for the files in the mbrl_traffic/utils/optimizers folder."""
import unittest

from mbrl_traffic.utils.optimizers.base import Optimizer
from mbrl_traffic.utils.optimizers import NelderMead
# from mbrl_traffic.utils.optimizers import GeneticAlgorithm
# from mbrl_traffic.utils.optimizers import CrossEntropyMethod


class TestOptimizer(unittest.TestCase):
    """Tests the Optimizer object."""

    def test_init(self):
        """Test that the base optimizer object is initialized properly."""
        optimizer = Optimizer(
            param_low=1,
            param_high=2,
            fitness_fn=3,
            verbose=4
        )
        self.assertEqual(optimizer.param_low, 1)
        self.assertEqual(optimizer.param_high, 2)
        self.assertEqual(optimizer.fitness_fn, 3)
        self.assertEqual(optimizer.verbose, 4)


class TestNelderMead(unittest.TestCase):
    """Tests the NelderMead object."""

    def test_init(self):
        """Check the functionality of the init method.

        This method checks that all attributes are properly initialized given
        the parameters that are passed.
        """
        # initialize the optimizer
        optimizer = NelderMead(
            x0=0,
            param_low=-1,
            param_high=1,
            fitness_fn=v_eq_max_function,
            verbose=2,
        )

        # check the attributes
        self.assertEqual(optimizer.x0, 0)
        self.assertEqual(optimizer.bnds, (-1, 1))
        self.assertEqual(optimizer.options, {
            'disp': True,
            'initial_simplex': None,
            'xatol': 1e-8,
            'fatol': 1000,
            'adaptive': False
        })

    def test_solve(self):
        """Checks the solve method for a 1-D problem.

        We test the solver on its ability to compute the equilibrium speed of a
        ring road with IDM vehicles with parameters specified in error fn.

        1. Problem: num_vehicles=22, length=230
           Expected solution: 3.71361481

        2. Problem: num_vehicles=22, length=260
           Expected solution: 5.13977943
        """
        optimizer = NelderMead(
            x0=[2.5],
            param_low=[0.1],
            param_high=[5.],
            fitness_fn=None,
            verbose=2,
        )

        # test case 1
        def fitness_fn(x):
            return v_eq_max_function(x, 22, 230)
        optimizer.fitness_fn = fitness_fn
        sol, err = optimizer.solve()
        self.assertAlmostEqual(sol[0], 3.71361481)

        # test case 2
        def fitness_fn(x):
            return v_eq_max_function(x, 22, 260)
        optimizer.fitness_fn = fitness_fn
        sol, err = optimizer.solve()
        self.assertAlmostEqual(sol[0], 5.13977943)


class TestGeneticAlgorithm(unittest.TestCase):
    """Tests the GeneticAlgorithm object."""

    def test_init(self):
        """Check the functionality of the init method.

        This method checks that all attributes are properly initialized given
        the parameters that are passed.
        """
        # initialize the optimizer
        optimizer = None  # TODO

        # check the attributes
        del optimizer  # TODO

        # initialize the optimizer with specified optional parameters
        optimizer = None  # TODO

        # check the attributes
        del optimizer  # TODO

    def test_solve(self):
        """Checks the solve method for a 1-D problem.

        We test the solver on its ability to compute the equilibrium speed of a
        ring road with IDM vehicles with parameters specified in error fn.

        1. Problem: num_vehicles=22, length=230
           Expected solution: 3.71361481

        2. Problem: num_vehicles=22, length=260
           Expected solution: 5.13977943
        """
        optimizer = None  # TODO
        del optimizer

        # test case 1  TODO
        # def fitness_fn(x):
        #     return v_eq_max_function(x, 22, 230)
        # optimizer.fitness_fn = fitness_fn
        # sol, err = optimizer.solve()
        # self.assertAlmostEqual(sol[0], 3.71361481)

        # test case 2  TODO
        # def fitness_fn(x):
        #     return v_eq_max_function(x, 22, 260)
        # optimizer.fitness_fn = fitness_fn
        # sol, err = optimizer.solve()
        # self.assertAlmostEqual(sol[0], 5.13977943)


class TestCrossEntropyMethod(unittest.TestCase):
    """Tests the CrossEntropyMethod object."""

    def test_init(self):
        """Check the functionality of the init method.

        This method checks that all attributes are properly initialized given
        the parameters that are passed.
        """
        # initialize the optimizer
        optimizer = None  # TODO

        # check the attributes
        del optimizer  # TODO

        # initialize the optimizer with specified optional parameters
        optimizer = None  # TODO

        # check the attributes
        del optimizer  # TODO

    def test_solve(self):
        """Checks the solve method for a 1-D problem.

        We test the solver on its ability to compute the equilibrium speed of a
        ring road with IDM vehicles with parameters specified in error fn.

        1. Problem: num_vehicles=22, length=230
           Expected solution: 3.71361481

        2. Problem: num_vehicles=22, length=260
           Expected solution: 5.13977943
        """
        optimizer = None  # TODO
        del optimizer

        # test case 1  TODO
        # def fitness_fn(x):
        #     return v_eq_max_function(x, 22, 230)
        # optimizer.fitness_fn = fitness_fn
        # sol, err = optimizer.solve()
        # self.assertAlmostEqual(sol[0], 3.71361481)

        # test case 2  TODO
        # def fitness_fn(x):
        #     return v_eq_max_function(x, 22, 260)
        # optimizer.fitness_fn = fitness_fn
        # sol, err = optimizer.solve()
        # self.assertAlmostEqual(sol[0], 5.13977943)


def v_eq_max_function(v, *args):
    """Return the error between the desired and actual equivalent gap.

    This is a test for computing the equilibrium speed of a ring road with IDM
    vehicles with parameters specified in the function below.
    """
    num_vehicles, length = args

    # maximum gap in the presence of one rl vehicle
    s_eq_max = (length - num_vehicles * 5) / (num_vehicles - 1)

    v0 = 30
    s0 = 2
    tau = 1
    gamma = 4

    error = (s_eq_max - (s0 + v * tau) * (1 - (v / v0) ** gamma) ** -0.5) ** 2

    return error


if __name__ == '__main__':
    unittest.main()
