"""Tests for the files in the mbrl_traffic/models folder."""
import unittest
import tensorflow as tf
from gym.spaces import Box

from mbrl_traffic.models.base import Model


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
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'verbose': 2,
        }


        replay_buffer,
        verbose,
        model_lr,
        layer_norm,
        layers,
        act_fun,
        stochastic,
        num_ensembles)

        FEEDFORWARD_PARAMS = dict(
        # the max number of transitions to store
        buffer_size = 200000,
        # the size of the batch for learning the policy
        batch_size = 128,
        # learning rate
        model_lr = 3e-4,
        # enable layer normalisation
        layer_norm = False,
        # the size of the neural network for the policy
        layers = [256, 256],
        # the activation function to use in the neural network
        act_fun = tf.nn.relu,
        # whether the output from the model is stochastic or deterministic
        use_huber = False,

    )

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


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
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
