"""Tests for the files in the mbrl_traffic/models folder."""
import unittest

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
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestNonLocalModel(unittest.TestCase):
    """Tests the NonLocalModel object."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
