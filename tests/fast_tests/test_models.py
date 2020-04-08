"""Tests for the files in the mbrl_traffic/models folder."""
import unittest
from gym.spaces import Box
import numpy as np
import tensorflow as tf

from mbrl_traffic.models.base import Model
from mbrl_traffic.models.no_op import NoOpModel
from mbrl_traffic.utils.replay_buffer import ReplayBuffer
from mbrl_traffic.models.fcnet import FeedForwardModel


class TestModel(unittest.TestCase):
    """Tests the Model object."""

    def test_init(self):
        """Test that the base model object is initialized properly."""
        model = Model(
            sess=1,
            ob_space=2,
            ac_space=3,
            replay_buffer=4,
            verbose=5
        )
        self.assertEqual(model.sess, 1)
        self.assertEqual(model.ob_space, 2)
        self.assertEqual(model.ac_space, 3)
        self.assertEqual(model.replay_buffer, 4)
        self.assertEqual(model.verbose, 5)


class TestARZModel(unittest.TestCase):
    """Tests the ARZModel object."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestFeedForwardModel(unittest.TestCase):
    """Tests the FeedForwardModel object."""

    def setUp(self):
        self.model_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'replay_buffer': ReplayBuffer(buffer_size=200000,
                                          batch_size=128,
                                          obs_dim=2,
                                          ac_dim=1),
            'verbose': 2
        }

        FEEDFORWARD_PARAMS = dict(  # TODO yf maybe move
            # learning rate
            model_lr=3e-4,
            # enable layer normalisation
            layer_norm=False,
            # the size of the neural network for the policy
            layers=[256, 256],
            # the activation function to use in the neural network
            act_fun=tf.nn.relu,
            # whether the output from the model is stochastic or deterministic
            stochastic=False,
            # number of ensembles
            num_ensembles=2
        )
        self.model_params.update(FEEDFORWARD_PARAMS.copy())
        self.model = FeedForwardModel(**self.model_params)

    def tearDown(self):
        self.model_params['sess'].close()
        del self.model_params
        del self.model

    def test_pass(self):
        """Check that the class has all the required attributes and methods.

        """

        # Check that the abstract class has all the required attributes.
        self.assertEqual(self.model.sess, self.model_params['sess'])
        self.assertEqual(self.model.ac_space, self.model_params['ac_space'])
        self.assertEqual(self.model.ob_space, self.model_params['ob_space'])
        self.assertEqual(self.model.replay_buffer,
                         self.model_params['replay_buffer'])
        self.assertEqual(self.model.verbose, self.model_params['verbose'])
        self.assertEqual(self.model.model_lr, self.model_params['model_lr'])
        self.assertEqual(self.model.layer_norm, self.model_params['layer_norm'])
        self.assertEqual(self.model.layers, self.model_params['layers'])
        self.assertEqual(self.model.act_fun, self.model_params['act_fun'])
        self.assertEqual(self.model.stochastic, self.model_params['stochastic'])
        self.assertEqual(self.model.num_ensembles,
                         self.model_params['num_ensembles'])

        # Check that the abstract class has all the required methods.
        self.assertRaises(NotImplementedError, self.model.initialize)
        self.assertRaises(NotImplementedError, self.model.get_next_obs,
                          obs=None, actions=None)
        self.assertRaises(NotImplementedError, self.model.update)
        self.assertRaises(NotImplementedError, self.model.log_loss,
                          y_true=None, mean=None, std=None)
        self.assertRaises(NotImplementedError, self.model.get_td_map)
        self.assertRaises(NotImplementedError, self.model.save,
                          save_path=None)
        self.assertRaises(NotImplementedError, self.model.load,
                          load_path=None)

        # test get_next_obs
        np.testing.assert_almost_equal(self.model.get_next_obs([], []), [0.])

        # test update
        self.assertEqual(self.model.update(), 0)

        # test compute_loss
        self.assertEqual(self.model.compute_loss([], [], []), 0)

        # test get_td_map
        self.assertEqual(self.model.get_td_map(), {})

        # test save
        self.model.save("")

        # test load
        self.model.load("")

    def test_log_loss(self):
        """Check the functionality of the log_loss() method.

        This method is tested for one cases:

        1. when standard deviation is zero and y_true is equal to the prediction
        means. in this case we expect the function return 1
        """
        # true value
        y_true = np.array([1, 0, 1])
        # predictions
        means = np.array([1, 0, 1])
        std = np.array([0, 0, 0])
        # expected
        expected = 1  # TODO check max_logstd clipping impact
        np.testing.assert_almost_equal(self.model.log_loss(y_true, means, std),
                                       expected)


class TestLWRModel(unittest.TestCase):
    """Tests the LWRModel object."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestNoOpModel(unittest.TestCase):
    """Tests the NoOpModel object."""

    def setUp(self):
        self.model = NoOpModel(
            sess=None,
            ob_space=Box(low=-1, high=1, shape=(1,)),
            ac_space=Box(low=-2, high=2, shape=(2,)),
            replay_buffer=None,
            verbose=2
        )

    def tearDown(self):
        del self.model

    def test_noop_model(self):
        """Check the functionality of the methods within this object."""
        # test initialize (this should just not fail)
        self.model.initialize()

        # test get_next_obs
        np.testing.assert_almost_equal(self.model.get_next_obs([], []), [0.])

        # test update
        self.assertEqual(self.model.update(), 0)

        # test compute_loss
        self.assertEqual(self.model.compute_loss([], [], []), 0)

        # test get_td_map
        self.assertEqual(self.model.get_td_map(), {})

        # test save (this should just not fail)
        self.model.save("")

        # test load (this should just not fail)
        self.model.load("")


if __name__ == '__main__':
    unittest.main()