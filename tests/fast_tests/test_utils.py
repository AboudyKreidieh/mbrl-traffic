"""TODO."""
import unittest
import numpy as np
import tensorflow as tf

from mbrl_traffic.utils.replay_buffer import ReplayBuffer
from mbrl_traffic.utils.tf_util import gaussian_likelihood


class TestMisc(unittest.TestCase):
    """Tests the methods in mbrl_traffic/utils/misc.py."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


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
        """Check the functionality of the gaussian_likelihood() method."""
        input_ = tf.constant([[0, 1, 2]], dtype=tf.float32)
        mu_ = tf.constant([[0, 0, 0]], dtype=tf.float32)
        log_std = tf.constant([[-4, -3, -2]], dtype=tf.float32)
        val = gaussian_likelihood(input_, mu_, log_std)
        expected = -304.65784

        self.assertAlmostEqual(self.sess.run(val)[0], expected, places=4)

    def test_apply_squashing(self):
        """Check the functionality of the apply_squashing() method."""
        pass  # TODO

    def test_layer(self):
        """Check the functionality of the layer() method."""
        pass  # TODO


class TestTrain(unittest.TestCase):
    """Tests the methods in mbrl_traffic/utils/train.py."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
