"""Tests for files in the experiments folder."""
import unittest
import os
import shutil

from experiments.simulate import parse_args as parse_args_simulate
from experiments.train_agent import main as train_agent


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

    def test_train_agent(self):
        """Test that the main method functions as expected."""
        # Run the script; verify it executes without failure.
        train_agent([
            "MountainCarContinuous-v0",
            "--total_steps", "2000",
            "--policy", "SACPolicy",
            "--model", "NoOpModel",
            "--initial_exploration_steps", "1",
        ])

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/SACPolicy-NoOpModel/MountainCarContinuous-v0")
        ))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))


class TestTrainModel(unittest.TestCase):
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
