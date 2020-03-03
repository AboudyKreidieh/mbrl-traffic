"""Tests for the files in the mbrl_traffic/policies folder."""
import unittest

from mbrl_traffic.policies.base import Policy


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
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestPPOPolicy(unittest.TestCase):
    """Tests the PPOPolicy object."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestSACPolicy(unittest.TestCase):
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
