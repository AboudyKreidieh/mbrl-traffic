"""Tests for files in the experiments folder."""
import unittest
import os
import shutil
import numpy as np

from mbrl_traffic.models import LWRModel
from mbrl_traffic.models import ARZModel
from mbrl_traffic.models import NoOpModel
from mbrl_traffic.models import FeedForwardModel

from experiments.evaluate_model import parse_args as parse_args_evaluate_model
from experiments.evaluate_model import import_data
from experiments.evaluate_model import get_model_cls
from experiments.evaluate_model import get_model_ckpt
from experiments.evaluate_model import main as evaluate_model
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

    def test_parse_args(self):
        """Validate the functionality of the parse_args() method."""
        # test the default case
        args = parse_args_evaluate_model(["."])
        self.assertDictEqual(args.__dict__, {
            'checkpoint_num': None,
            'initial_conditions': None,
            'plot_only': False,
            'results_dir': '.',
            'runs': 1,
            'save_path': None,
            'steps': 1000,
            'svg': False
        })

        # test the updated case
        args = parse_args_evaluate_model([
            '.',
            '--checkpoint_num', '1',
            '--initial_conditions', 'initial_conditions',
            '--plot_only',
            '--runs', '2',
            '--save_path', 'save_path',
            '--steps', '3',
            '--svg',
        ])
        self.assertDictEqual(args.__dict__, {
            'checkpoint_num': 1,
            'initial_conditions': 'initial_conditions',
            'plot_only': True,
            'results_dir': '.',
            'runs': 2,
            'save_path': 'save_path',
            'steps': 3,
            'svg': True
        })

    def test_import_data(self):
        """Validate the functionality of the import_data() method.

        We test this method on a small csv file that is representative of what
        the method is meant to import.
        """
        times, obses = import_data(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data/sample_macro.csv"))

        np.testing.assert_almost_equal(times, [0, 0.5, 1])
        np.testing.assert_almost_equal(obses, [[1,  2,  3,  4],
                                               [5,  6,  7,  8],
                                               [9, 10, 11, 12]])

    def test_get_model_cls(self):
        """Validate the functionality of the get_model_cls() method.

        This method is tested for the following cases:

        1. model = "LWRModel"
        2. model = "ARZModel"
        3. model = "NonLocalModel"
        4. model = "NoOpModel"
        5. model = "FeedForwardModel"
        6. model = "dns dl"  <-- returns a ValueError
        """
        # test case 1
        self.assertEqual(get_model_cls("LWRModel"), LWRModel)

        # test case 2
        self.assertEqual(get_model_cls("ARZModel"), ARZModel)

        # test case 3  TODO
        # self.assertEqual(get_model_cls("NonLocalModel"), None)

        # test case 4
        self.assertEqual(get_model_cls("NoOpModel"), NoOpModel)

        # test case 5
        self.assertEqual(get_model_cls("FeedForwardModel"), FeedForwardModel)

        # test case 6
        self.assertRaises(ValueError, get_model_cls, model="dns dl")

    def test_get_model_ckpt(self):
        """Validate the functionality of the get_model_ckpt() method.

        This method is tested for the following cases:

        1. model = "NoOpModel"
        2. model = "FeedForwardModel"
        3. model = "LWRModel", ckpt_num = None
        4. model = "LWRModel", ckpt_num = 1
        """
        # test case 1
        self.assertEqual(get_model_ckpt("base_dir", 1, "NoOpModel"), None)

        # test case 2
        self.assertEqual(get_model_ckpt("base_dir", 1, "FeedForwardModel"),
                         "base_dir/checkpoints/itr-1/model.meta")

        # test case 3
        self.assertEqual(
            get_model_ckpt(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "data/sample_log"
                ),
                None,
                "LWRModel"
            ),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data/sample_log/checkpoints/itr-2/model.csv"
            )
        )

        # test case 4
        self.assertEqual(get_model_ckpt("base_dir", 1, "LWRModel"),
                         "base_dir/checkpoints/itr-1/model.csv")

    def test_main_plot_only(self):
        """Validate the functionality of the main() method when plotting."""
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        evaluate_model([os.path.join(cur_dir, "data"), "--plot_only"])

        # Check that the necessary files are generated.
        self.assertTrue(os.path.isfile(
            os.path.join(cur_dir, "data/speed_0.png")))
        self.assertTrue(os.path.isfile(
            os.path.join(cur_dir, "data/density_0.png")))
        self.assertTrue(os.path.isfile(
            os.path.join(cur_dir, "data/flow_0.png")))
        self.assertTrue(os.path.isfile(
            os.path.join(cur_dir, "data/speed_0.mp4")))
        self.assertTrue(os.path.isfile(
            os.path.join(cur_dir, "data/density_0.mp4")))

        # Check the generated files.
        os.remove(os.path.join(cur_dir, "data/speed_0.png"))
        os.remove(os.path.join(cur_dir, "data/density_0.png"))
        os.remove(os.path.join(cur_dir, "data/flow_0.png"))
        os.remove(os.path.join(cur_dir, "data/speed_0.mp4"))
        os.remove(os.path.join(cur_dir, "data/density_0.mp4"))

    def test_main_evaluate(self):
        """Validate the functionality of the main() method when evaluating."""
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
