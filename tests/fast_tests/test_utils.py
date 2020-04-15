"""Tests for the files in the mbrl_traffic/utils folder."""
import unittest
import numpy as np
import tensorflow as tf

from mbrl_traffic.utils.replay_buffer import ReplayBuffer
from mbrl_traffic.utils.tf_util import gaussian_likelihood
from mbrl_traffic.utils.train import create_env
from mbrl_traffic.utils.train import parse_params
# from mbrl_traffic.utils.train import get_algorithm_params_from_args
# from mbrl_traffic.utils.train import get_policy_params_from_args
# from mbrl_traffic.utils.train import get_model_params_from_args


class TestReplayBuffer(unittest.TestCase):
    """Tests the ReplayBuffer object."""

    def setUp(self):
        self.replay_buffer = ReplayBuffer(
            buffer_size=2, batch_size=1, obs_dim=1, ac_dim=1)

    def tearDown(self):
        del self.replay_buffer

    def test_buffer_size(self):
        """Validate the buffer_size output from the replay buffer."""
        self.assertEqual(self.replay_buffer.buffer_size, 2)

    def test_add_sample(self):
        """Test the `add` and `sample` methods the replay buffer."""
        # Add an element.
        self.replay_buffer.add(
            obs_t=np.array([0]),
            action=np.array([1]),
            reward=2,
            obs_tp1=np.array([3]),
            done=False
        )

        # Check is_full in the False case.
        self.assertEqual(self.replay_buffer.is_full(), False)

        # Add an element.
        self.replay_buffer.add(
            obs_t=np.array([0]),
            action=np.array([1]),
            reward=2,
            obs_tp1=np.array([3]),
            done=False
        )

        # Check is_full in the True case.
        self.assertEqual(self.replay_buffer.is_full(), True)

        # Check can_sample in the True case.
        self.assertEqual(self.replay_buffer.can_sample(), True)

        # Test the `sample` method.
        obs_t, actions_t, rewards, obs_tp1, done = self.replay_buffer.sample()
        np.testing.assert_array_almost_equal(obs_t, [[0]])
        np.testing.assert_array_almost_equal(actions_t, [[1]])
        np.testing.assert_array_almost_equal(rewards, [2])
        np.testing.assert_array_almost_equal(obs_tp1, [[3]])
        np.testing.assert_array_almost_equal(done, [False])


class TestTfUtil(unittest.TestCase):
    """Tests the methods in mbrl_traffic/utils/tf_util.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

    def tearDown(self):
        self.sess.close()

    def test_gaussian_likelihood(self):
        """Check the functionality of the gaussian_likelihood method."""
        input_ = tf.constant([[0, 1, 2]], dtype=tf.float32)
        mu_ = tf.constant([[0, 0, 0]], dtype=tf.float32)
        log_std = tf.constant([[-4, -3, -2]], dtype=tf.float32)
        val = gaussian_likelihood(input_, mu_, log_std)
        expected = -304.65784

        self.assertAlmostEqual(self.sess.run(val)[0], expected, places=4)

    def test_apply_squashing(self):
        """Check the functionality of the apply_squashing method."""
        pass  # TODO

    def test_layer(self):
        """Check the functionality of the _layer method.

        This method is tested for the following features:

        1. the number of outputs from the layer equals num_outputs
        2. the name is properly used
        3. the proper activation function applied if requested
        4. weights match what the kernel_initializer requests (tested on a
           constant initializer)
        5. layer_norm is applied if requested
        """
        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

        # test case 4
        pass  # TODO

        # test case 5
        pass  # TODO


class TestTrain(unittest.TestCase):
    """Tests the methods in mbrl_traffic/utils/train.py."""

    def test_create_env(self):
        """Check the functionality of the create_env method.

        This method checks that environments with the proper space and action
        spaces are generated under the possible cases:

        1. MountainCarContinuous-v0
        2. av-ring
        3. av-merge
        4. av-highway
        5. vsl-ring
        6. vsl-merge
        7. vsl-highway
        """
        # test case 1
        env = create_env("MountainCarContinuous-v0")
        test_space(
            gym_space=env.observation_space,
            expected_size=2,
            expected_min=np.array([-1.2, -0.07]),
            expected_max=np.array([0.6, 0.07]),
        )
        test_space(
            gym_space=env.action_space,
            expected_size=1,
            expected_min=-1,
            expected_max=1,
        )

        # test case 2
        env = create_env("av-ring")
        test_space(
            gym_space=env.observation_space,
            expected_size=25,
            expected_min=-float("inf"),
            expected_max=float("inf"),
        )
        test_space(
            gym_space=env.action_space,
            expected_size=5,
            expected_min=-1,
            expected_max=1,
        )

        # test case 3
        pass  # TODO

        # test case 4
        pass  # TODO

        # test case 5
        pass  # TODO

        # test case 6
        pass  # TODO

        # test case 7
        pass  # TODO

    def test_parse_params(self):
        """Check the functionality of the parse_params method.

        This method tests the relevant arguments for the following categories:

        1. no specific sub-category
        2. parse_algorithm_params
        3. parse_policy_params
        4. parse_model_params
        """
        # test case (default)
        args = parse_params(["vsl-ring"])

        self.assertEqual(args.__dict__, {
            'env_name': 'vsl-ring',
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'evaluate': False,
            'nb_eval_episodes': 50,
            'policy_update_freq': 1,
            'model_update_freq': 10,
            'buffer_size': 200000,
            'batch_size': 128,
            'reward_scale': 1.0,
            'model_reward_scale': 1.0,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'policy': 'SACPolicy',
            'num_particles': 1,
            'traj_length': 10,
            'actor_lr': 0.0003,
            'critic_lr': 0.0003,
            'tau': 0.005,
            'gamma': 0.99,
            'layer_norm': False,
            'use_huber': False,
            'target_entropy': None,
            'model': 'NoOpModel',
            'optimizer_cls': 'GeneticAlgorithm',
            'dx': 50,
            'dt': 0.5,
            'rho_max': None,
            'rho_max_max': 1,
            'v_max': None,
            'v_max_max': 30,
            'lam': None,
            'boundary_conditions': 'loop',
            'stream_model': None,
            'tau_arz': None,
            'model_lr': 1e-05,
            'layer_norm_model': False,
            'stochastic': False,
            'num_ensembles': 1
        })

        # test case (pre-defined)
        args = parse_params([
            'vsl-ring',
            '--n_training', '1',
            '--total_steps', '2',
            '--seed', '3',
            '--log_interval', '4',
            '--eval_interval', '5',
            '--save_interval', '6',
            '--initial_exploration_steps', '7',
            '--evaluate',
            '--nb_eval_episodes', '8',
            '--policy_update_freq', '9',
            '--model_update_freq', '10',
            '--buffer_size', '11',
            '--batch_size', '12',
            '--reward_scale', '13.0',
            '--model_reward_scale', '14.0',
            '--render',
            '--render_eval',
            '--verbose', '15',
            '--policy', '16',
            '--num_particles', '17',
            '--traj_length', '18',
            '--actor_lr', '19.0',
            '--critic_lr', '20.0',
            '--tau', '21.0',
            '--gamma', '22.0',
            '--layer_norm',
            '--use_huber',
            '--target_entropy', '23',
            '--model', '24',
            '--optimizer_cls', '25',
            '--dx', '26',
            '--dt', '27.0',
            '--rho_max', '28',
            '--rho_max_max', '29',
            '--v_max', '30',
            '--v_max_max', '31',
            '--lam', '32',
            '--boundary_conditions', '33',
            '--stream_model', '34',
            '--tau_arz', '35',
            '--model_lr', '36',
            '--layer_norm_model',
            '--stochastic',
            '--num_ensembles', '37',
        ])
        self.maxDiff = None
        self.assertEqual(args.__dict__, {
            'env_name': 'vsl-ring',
            'n_training': 1,
            'total_steps': 2,
            'seed': 3,
            'log_interval': 4,
            'eval_interval': 5,
            'save_interval': 6,
            'initial_exploration_steps': 7,
            'evaluate': True,
            'nb_eval_episodes': 8,
            'policy_update_freq': 9,
            'model_update_freq': 10,
            'buffer_size': 11,
            'batch_size': 12,
            'reward_scale': 13.0,
            'model_reward_scale': 14.0,
            'render': True,
            'render_eval': True,
            'verbose': 15,
            'policy': '16',
            'num_particles': 17,
            'traj_length': 18,
            'actor_lr': 19.0,
            'critic_lr': 20.0,
            'tau': 21.0,
            'gamma': 22.0,
            'layer_norm': True,
            'use_huber': True,
            'target_entropy': 23.0,
            'model': '24',
            'optimizer_cls': '25',
            'dx': 26.0,
            'dt': 27.0,
            'rho_max': 28.0,
            'rho_max_max': 29.0,
            'v_max': 30.0,
            'v_max_max': 31.0,
            'lam': 32,
            'boundary_conditions': '33',
            'stream_model': '34',
            'tau_arz': 35.0,
            'model_lr': 36.0,
            'layer_norm_model': True,
            'stochastic': True,
            'num_ensembles': 37
        })

    def test_get_algorithm_params_from_args(self):
        """Check the validity of the get_algorithm_params_from_args method.

        This method tests that the the output algorithm parameters match their
        expected values.
        """
        pass  # TODO

    def test_get_policy_params_from_args(self):
        """Check the validity of the get_policy_params_from_args method.

        This method tests the output policy class and policy parameters under
        the following cases:

        1. using NoOpPolicy
        2. using SACPolicy
        3. using KShootPolicy
        """
        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

    def test_get_model_params_from_args(self):
        """Check the validity of the get_model_params_from_args method.

        This method tests the output model class and model parameters under the
        following cases:

        1. using ARZModel
        2. using LWRModel
        3. using FeedForwardModel
        4. using NoOpModel
        """
        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

        # test case 4
        pass  # TODO


def test_space(gym_space, expected_size, expected_min, expected_max):
    """Test the shape and bounds of an action or observation space.

    Parameters
    ----------
    gym_space : gym.spaces.Box
        gym space object to be tested
    expected_size : int
        expected size
    expected_min : float or array_like
        expected minimum value(s)
    expected_max : float or array_like
        expected maximum value(s)
    """
    assert gym_space.shape[0] == expected_size, \
        "{}, {}".format(gym_space.shape[0], expected_size)
    np.testing.assert_almost_equal(gym_space.high, expected_max, decimal=4)
    np.testing.assert_almost_equal(gym_space.low, expected_min, decimal=4)


if __name__ == '__main__':
    unittest.main()
