"""Tests for the files in the mbrl_traffic/policies folder."""
import unittest
from gym.spaces import Box
import numpy as np

from mbrl_traffic.policies.base import Policy
from mbrl_traffic.policies import NoOpPolicy


class TestPolicy(unittest.TestCase):
    """Tests the Policy object."""

    def test_init(self):
        """Test that the base policy object is initialized properly."""
        policy = Policy(
            sess=1,
            ob_space=2,
            ac_space=3,
            model=4,
            replay_buffer=5,
            verbose=6
        )
        self.assertEqual(policy.sess, 1)
        self.assertEqual(policy.ob_space, 2)
        self.assertEqual(policy.ac_space, 3)
        self.assertEqual(policy.model, 4)
        self.assertEqual(policy.replay_buffer, 5)
        self.assertEqual(policy.verbose, 6)


class TestKShootPolicy(unittest.TestCase):
    """Tests the KShootPolicy object."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestNoOpPolicy(unittest.TestCase):
    """Tests the NoOpPolicy object."""

    def setUp(self):
        self.policy = NoOpPolicy(
            sess=None,
            ob_space=Box(low=-1, high=1, shape=(1,)),
            ac_space=Box(low=-2, high=2, shape=(2,)),
            model=None,
            replay_buffer=None,
            verbose=2,
        )

    def tearDown(self):
        del self.policy

    def test_pass(self):
        """Check the functionality of the methods within this object."""
        # test initialize (this should just not fail)
        self.policy.initialize()

        # test update
        loss1, loss2, loss3 = self.policy.update()
        self.assertEqual(loss1, 0)
        self.assertEqual(loss2, (0,))
        self.assertEqual(loss3, {})

        # test get_action
        np.random.seed(0)
        np.testing.assert_almost_equal(
            self.policy.get_action([], False, False),
            [0.195254, 0.8607575]
        )

        # test value
        self.assertEqual(self.policy.value([], []), (0,))

        # test get_td_map
        self.assertEqual(self.policy.get_td_map(), {})

        # test save (this should just not fail)
        self.policy.save("")

        # test load (this should just not fail)
        self.policy.load("")


class TestSACPolicy(unittest.TestCase):  # TODO
    """Tests the SACPolicy object."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
