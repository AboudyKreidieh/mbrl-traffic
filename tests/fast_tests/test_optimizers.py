"""Tests for the files in the mbrl_traffic/utils/optimizers folder."""
import unittest

from mbrl_traffic.utils.optimizers.base import Optimizer


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

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestGeneticAlgorithm(unittest.TestCase):  # TODO
    """Tests the GeneticAlgorithm object."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_selection(self):
        """TODO."""
        pass  # TODO

    def test_pairing(self):
        """TODO."""
        pass  # TODO

    def test_mating(self):
        """TODO."""
        pass  # TODO

    def test_mutation(self):
        """TODO."""
        pass  # TODO

    def test_case1(self):
        """TODO."""
        pass  # TODO

    def test_case2(self):
        """TODO."""
        pass  # TODO


class TestCrossEntropyMethod(unittest.TestCase):
    """Tests the CrossEntropyMethod object."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
