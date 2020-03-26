"""Tests for the files in the mbrl_traffic/policies folder."""
import unittest
from gym.spaces import Box
import numpy as np
import tensorflow as tf

from mbrl_traffic.policies.base import Policy
from mbrl_traffic.policies import NoOpPolicy
from mbrl_traffic.policies import SACPolicy
from mbrl_traffic.utils.train import SAC_POLICY_PARAMS
from mbrl_traffic.utils.tf_util import get_trainable_vars


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

    def test_policy(self):
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
            [0.8607575, 0.4110535]
        )

        # test value
        self.assertEqual(self.policy.value([], []), (0,))

        # test get_td_map
        self.assertEqual(self.policy.get_td_map(), {})

        # test save (this should just not fail)
        self.policy.save("")

        # test load (this should just not fail)
        self.policy.load("")


class TestSACPolicy(unittest.TestCase):
    """Tests the SACPolicy object."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'model': NoOpPolicy,
            'replay_buffer': None,
            'verbose': 2,
        }
        self.policy_params.update(SAC_POLICY_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.
        3. self.log_alpha is initialized to zero
        4. self.target_entropy is initialized as specified, with the special
           (None) case as well
        """
        with tf.compat.v1.variable_scope('policy'):
            policy = SACPolicy(**self.policy_params)

            # test case 1
            self.assertListEqual(
                sorted([var.name for var in get_trainable_vars()]),
                ['policy/model/log_alpha:0',
                 'policy/model/pi/fc0/bias:0',
                 'policy/model/pi/fc0/kernel:0',
                 'policy/model/pi/fc1/bias:0',
                 'policy/model/pi/fc1/kernel:0',
                 'policy/model/pi/log_std/bias:0',
                 'policy/model/pi/log_std/kernel:0',
                 'policy/model/pi/mean/bias:0',
                 'policy/model/pi/mean/kernel:0',
                 'policy/model/value_fns/qf1/fc0/bias:0',
                 'policy/model/value_fns/qf1/fc0/kernel:0',
                 'policy/model/value_fns/qf1/fc1/bias:0',
                 'policy/model/value_fns/qf1/fc1/kernel:0',
                 'policy/model/value_fns/qf1/qf_output/bias:0',
                 'policy/model/value_fns/qf1/qf_output/kernel:0',
                 'policy/model/value_fns/qf2/fc0/bias:0',
                 'policy/model/value_fns/qf2/fc0/kernel:0',
                 'policy/model/value_fns/qf2/fc1/bias:0',
                 'policy/model/value_fns/qf2/fc1/kernel:0',
                 'policy/model/value_fns/qf2/qf_output/bias:0',
                 'policy/model/value_fns/qf2/qf_output/kernel:0',
                 'policy/model/value_fns/vf/fc0/bias:0',
                 'policy/model/value_fns/vf/fc0/kernel:0',
                 'policy/model/value_fns/vf/fc1/bias:0',
                 'policy/model/value_fns/vf/fc1/kernel:0',
                 'policy/model/value_fns/vf/vf_output/bias:0',
                 'policy/model/value_fns/vf/vf_output/kernel:0',
                 'policy/target/value_fns/vf/fc0/bias:0',
                 'policy/target/value_fns/vf/fc0/kernel:0',
                 'policy/target/value_fns/vf/fc1/bias:0',
                 'policy/target/value_fns/vf/fc1/kernel:0',
                 'policy/target/value_fns/vf/vf_output/bias:0',
                 'policy/target/value_fns/vf/vf_output/kernel:0']
            )

            # test case 2
            self.assertEqual(
                tuple(v.__int__() for v in policy.terminals1.shape),
                (None, 1))
            self.assertEqual(
                tuple(v.__int__() for v in policy.rew_ph.shape),
                (None, 1))
            self.assertEqual(
                tuple(v.__int__() for v in policy.action_ph.shape),
                (None, self.policy_params['ac_space'].shape[0]))
            self.assertEqual(
                tuple(v.__int__() for v in policy.obs_ph.shape),
                (None, self.policy_params['ob_space'].shape[0]))
            self.assertEqual(
                tuple(v.__int__() for v in policy.obs1_ph.shape),
                (None, self.policy_params['ob_space'].shape[0]))

            # Initialize the variables of the policy.
            policy.sess.run(tf.compat.v1.global_variables_initializer())

            # test case 3
            self.assertEqual(policy.sess.run(policy.log_alpha), 0.0)

            # test case 4a
            self.assertEqual(policy.target_entropy,
                             -self.policy_params['ac_space'].shape[0])

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.variable_scope('policy'):
            # test case 4b
            self.policy_params['target_entropy'] = 5
            policy = SACPolicy(**self.policy_params)
            self.assertEqual(policy.target_entropy,
                             self.policy_params['target_entropy'])

    def test_initialize(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.
        """
        with tf.compat.v1.variable_scope('policy'):
            policy = SACPolicy(**self.policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'policy/model/value_fns/vf/fc0/kernel:0',
            'policy/model/value_fns/vf/fc0/bias:0',
            'policy/model/value_fns/vf/fc1/kernel:0',
            'policy/model/value_fns/vf/fc1/bias:0',
            'policy/model/value_fns/vf/vf_output/kernel:0',
            'policy/model/value_fns/vf/vf_output/bias:0',
        ]

        target_var_list = [
            'policy/target/value_fns/vf/fc0/kernel:0',
            'policy/target/value_fns/vf/fc0/bias:0',
            'policy/target/value_fns/vf/fc1/kernel:0',
            'policy/target/value_fns/vf/fc1/bias:0',
            'policy/target/value_fns/vf/vf_output/kernel:0',
            'policy/target/value_fns/vf/vf_output/bias:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)


if __name__ == '__main__':
    unittest.main()
