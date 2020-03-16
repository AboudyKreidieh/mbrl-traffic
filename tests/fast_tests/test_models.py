"""Tests for the files in the mbrl_traffic/models folder."""
import unittest
from gym.spaces import Box
import numpy as np

from mbrl_traffic.models.base import Model
from mbrl_traffic.models.no_op import NoOpModel


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
        pass  # TODO

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
