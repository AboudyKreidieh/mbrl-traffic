"""Tests for files in the experiments folder."""
import unittest

from experiments.simulate import parse_args as parse_args_simulate


class TestEvaluateAgent(unittest.TestCase):
    """Tests the experiments/evaluate_agent.py script."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestEvaluateModel(unittest.TestCase):
    """Tests the experiments/evaluate_model.py script."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestSimulate(unittest.TestCase):
    """Tests the experiments/simulate.py script."""

    def test_parse_args(self):
        """Validate the functionality of the parse_args() method."""
        # test the default case
        args = parse_args_simulate(["merge"])
        self.assertDictEqual(args.__dict__, {
            'exp_config': 'merge',
            'num_runs': 1,
            'no_render': False,
            'aimsun': False,
            'gen_emission': False,
            'ring_length': 1500,
            'ring_lanes': 1,
        })

        # test the updated case
        args = parse_args_simulate([
            'merge',
            '--num_runs', '2',
            '--no_render',
            '--aimsun',
            '--gen_emission',
            '--ring_length', '3',
            '--ring_lanes', '4',
        ])
        self.assertDictEqual(args.__dict__, {
            'exp_config': 'merge',
            'num_runs': 2,
            'no_render': True,
            'aimsun': True,
            'gen_emission': True,
            'ring_length': 3,
            'ring_lanes': 4,
        })

    def test_run_ring(self):
        """TODO."""
        pass  # TODO

    def test_run_merge(self):
        """TODO."""
        pass  # TODO

    def test_run_highway(self):
        """TODO."""
        pass  # TODO


class TestTrainAgent(unittest.TestCase):
    """Tests the experiments/train_agent.py script."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestTrainModel(unittest.TestCase):  # TODO
    """Tests the experiments/train_model.py script."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
