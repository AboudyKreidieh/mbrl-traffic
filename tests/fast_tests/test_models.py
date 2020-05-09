"""Tests for the files in the mbrl_traffic/models folder."""
import unittest
from gym.spaces import Box
import numpy as np
import os
import json
from copy import deepcopy
import shutil
import tensorflow as tf

from mbrl_traffic.models.base import Model
from mbrl_traffic.models import LWRModel
from mbrl_traffic.models import ARZModel
from mbrl_traffic.models import NoOpModel
from mbrl_traffic.utils.replay_buffer import ReplayBuffer
from mbrl_traffic.utils.optimizers import NelderMead
from mbrl_traffic.utils.train import LWR_MODEL_PARAMS
from mbrl_traffic.utils.train import ARZ_MODEL_PARAMS
from mbrl_traffic.models.fcnet import FeedForwardModel
from mbrl_traffic.utils.train import FEEDFORWARD_MODEL_PARAMS


class TestModel(unittest.TestCase):
    """Tests the Model object."""

    def test_init(self):
        """Test that the base model object is initialized properly."""
        model = Model(
            sess=1,
            ob_space=2,
            ac_space=3,
            replay_buffer=ReplayBuffer(1, 1, 2, 3),
            verbose=5
        )
        self.assertEqual(model.sess, 1)
        self.assertEqual(model.ob_space, 2)
        self.assertEqual(model.ac_space, 3)
        self.assertEqual(model.replay_buffer.obs_dim, 2)
        self.assertEqual(model.replay_buffer.ac_dim, 3)
        self.assertEqual(model.verbose, 5)


class TestARZModel(unittest.TestCase):
    """Tests the ARZModel object."""

    def setUp(self):
        self.model_cls = ARZModel
        self.model_params = deepcopy(ARZ_MODEL_PARAMS)
        self.model_params.update({
            "sess": 1,
            "ob_space": 2,
            "ac_space": 3,
            "replay_buffer": 4,
            "verbose": 5,
            "optimizer_cls": NelderMead,
        })

    def test_init(self):
        """Validate the functionality of the __init__ method.

        This method check that the attributes are properly initialized.
        """
        model = self.model_cls(**self.model_params)
        self.assertEqual(model.dx, ARZ_MODEL_PARAMS["dx"])
        self.assertEqual(model.dt, ARZ_MODEL_PARAMS["dt"])
        self.assertEqual(model.rho_max, ARZ_MODEL_PARAMS["rho_max"])
        self.assertEqual(model.rho_max_max, ARZ_MODEL_PARAMS["rho_max_max"])
        self.assertEqual(model.v_max, ARZ_MODEL_PARAMS["v_max"])
        self.assertEqual(model.v_max_max, ARZ_MODEL_PARAMS["v_max_max"])
        self.assertEqual(model.tau, ARZ_MODEL_PARAMS["tau"])
        self.assertEqual(model.tau_max, ARZ_MODEL_PARAMS["tau_max"])
        self.assertEqual(model.lam, ARZ_MODEL_PARAMS["lam"])
        self.assertEqual(model.boundary_conditions,
                         ARZ_MODEL_PARAMS["boundary_conditions"])
        self.assertEqual(model.optimizer_cls, NelderMead)
        self.assertEqual(model.optimizer_params, {})

    def test_get_next_obs(self):
        """Validate the functionality of the get_next_obs method.

        We check that the correct value mathematically is returned.This is
        done for the following boundary conditions:

        1. loop
        2. extend_both
        2. constant_both, [0, 0]
        """
        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

    def test_compute_loss(self):
        """Validate the functionality of the compute_loss method.

        We check that the correct value mathematically is returned.
        """
        pass  # TODO

    def test_save(self):
        """Validate the functionality of the save method.

        This method tries to save a set of pre-defined model attributes and
        ensures that the parameters are properly saved.
        """
        model = self.model_cls(**self.model_params)

        # Update the attributes.
        model.dx = 1
        model.dt = 2
        model.rho_max = 3
        model.rho_max_max = 4
        model.v_max = 5
        model.v_max_max = 6
        model.tau = 7
        model.tau_max = 8
        model.lam = 9
        model.boundary_conditions = 10

        # Save the attributes.
        model.save(".")

        # Check that the correct values were saved.
        with open('model.json') as f:
            params = json.load(f)
        self.assertEqual(params["dx"], 1)
        self.assertEqual(params["dt"], 2)
        self.assertEqual(params["rho_max"], 3)
        self.assertEqual(params["rho_max_max"], 4)
        self.assertEqual(params["v_max"], 5)
        self.assertEqual(params["v_max_max"], 6)
        self.assertEqual(params["tau"], 7)
        self.assertEqual(params["tau_max"], 8)
        self.assertEqual(params["lam"], 9)
        self.assertEqual(params["boundary_conditions"], 10)

        # Delete the generated json file.
        os.remove("model.json")

    def test_load(self):
        """Validate the functionality of the load method.

        This test imports a pre-defined saved set of parameters and ensures
        that the attributes of the model are accordingly modified.
        """
        model = self.model_cls(**self.model_params)

        # Load attributes from a pre-defined json file.
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model.load(os.path.join(cur_dir, "data/arz_model.json"))

        # Check that the correct values were loaded.
        self.assertEqual(model.dx, 1)
        self.assertEqual(model.dt, 2)
        self.assertEqual(model.rho_max, 3)
        self.assertEqual(model.rho_max_max, 4)
        self.assertEqual(model.v_max, 5)
        self.assertEqual(model.v_max_max, 6)
        self.assertEqual(model.tau, 7)
        self.assertEqual(model.tau_max, 8)
        self.assertEqual(model.lam, 9)
        self.assertEqual(model.boundary_conditions, 10)


class TestFeedForwardModel(unittest.TestCase):  # TODO
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

        self.model_params.update(FEEDFORWARD_MODEL_PARAMS.copy())
        self.model = FeedForwardModel(**self.model_params)
        self.model.sess.run(tf.compat.v1.global_variables_initializer())

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

        # test get_td_map
        self.assertEqual(self.model.get_td_map(), {})

        # test save
        self.model.save("./ckpts")

        # test load
        self.model.load("./ckpts")
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        shutil.rmtree(os.path.join(cur_dir, "ckpts"))

    def test_log_loss(self):
        """Check the functionality of the log_loss() method.

        This method is tested for one cases:

        1. when standard deviation is zero and y_true is equal to the prediction
        means. in this case we expect the function return 0
        """
        # true value
        y_true = tf.constant([1, 0, 1], dtype=tf.float32)

        # predictions
        means = tf.constant([1, 0, 1], dtype=tf.float32)
        std = tf.constant([1.e-16, 1.e-16, 1.e-16], dtype=tf.float32)

        # calculated loss value
        loss = self.model.log_loss(y_true, means, std)
        loss_value = self.model.sess.run(loss)

        # expected
        expected = 0.  # TODO check max_logstd clipping impact, check value

        # np.testing.assert_almost_equal(loss_value, expected)


class TestLWRModel(unittest.TestCase):
    """Tests the LWRModel object."""

    def setUp(self):
        self.model_cls = LWRModel
        self.model_params = deepcopy(LWR_MODEL_PARAMS)
        self.model_params.update({
            "sess": 1,
            "ob_space": 2,
            "ac_space": 3,
            "replay_buffer": 4,
            "verbose": 5,
            "optimizer_cls": NelderMead,
        })

    def test_init(self):
        """Validate the functionality of the __init__ method.

        This method check that the attributes are properly initialized.
        """
        model = self.model_cls(**self.model_params)
        self.assertEqual(model.dx, LWR_MODEL_PARAMS["dx"])
        self.assertEqual(model.dt, LWR_MODEL_PARAMS["dt"])
        self.assertEqual(model.rho_max, LWR_MODEL_PARAMS["rho_max"])
        self.assertEqual(model.rho_max_max, LWR_MODEL_PARAMS["rho_max_max"])
        self.assertEqual(model.v_max, LWR_MODEL_PARAMS["v_max"])
        self.assertEqual(model.v_max_max, LWR_MODEL_PARAMS["v_max_max"])
        self.assertEqual(model.stream_model, LWR_MODEL_PARAMS["stream_model"])
        self.assertEqual(model.lam, LWR_MODEL_PARAMS["lam"])
        self.assertEqual(model.boundary_conditions,
                         LWR_MODEL_PARAMS["boundary_conditions"])
        self.assertEqual(model.optimizer_cls, NelderMead)
        self.assertEqual(model.optimizer_params, {})

    def test_get_next_obs(self):
        """Validate the functionality of the get_next_obs method.

        We check that the correct value mathematically is returned.This is
        done for the following boundary conditions:

        1. loop
        2. extend_both
        2. constant_both, [0, 0]
        """
        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

    def test_compute_loss(self):
        """Validate the functionality of the compute_loss method.

        We check that the correct value mathematically is returned.
        """
        pass  # TODO

    def test_save(self):
        """Validate the functionality of the save method.

        This method tries to save a set of pre-defined model attributes and
        ensures that the parameters are properly saved.
        """
        model = self.model_cls(**self.model_params)

        # Update the attributes.
        model.dx = 1
        model.dt = 2
        model.rho_max = 3
        model.rho_max_max = 4
        model.v_max = 5
        model.v_max_max = 6
        model.stream_model = 7
        model.lam = 8
        model.boundary_conditions = 9

        # Save the attributes.
        model.save(".")

        # Check that the correct values were saved.
        with open('model.json') as f:
            params = json.load(f)
        self.assertEqual(params["dx"], 1)
        self.assertEqual(params["dt"], 2)
        self.assertEqual(params["rho_max"], 3)
        self.assertEqual(params["rho_max_max"], 4)
        self.assertEqual(params["v_max"], 5)
        self.assertEqual(params["v_max_max"], 6)
        self.assertEqual(params["stream_model"], 7)
        self.assertEqual(params["lam"], 8)
        self.assertEqual(params["boundary_conditions"], 9)

        # Delete the generated json file.
        os.remove("model.json")

    def test_load(self):
        """Validate the functionality of the load method.

        This test imports a pre-defined saved set of parameters and ensures
        that the attributes of the model are accordingly modified.
        """
        model = self.model_cls(**self.model_params)

        # Load attributes from a pre-defined json file.
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model.load(os.path.join(cur_dir, "data/lwr_model.json"))

        # Check that the correct values were loaded.
        self.assertEqual(model.dx, 1)
        self.assertEqual(model.dt, 2)
        self.assertEqual(model.rho_max, 3)
        self.assertEqual(model.rho_max_max, 4)
        self.assertEqual(model.v_max, 5)
        self.assertEqual(model.v_max_max, 6)
        self.assertEqual(model.stream_model, 7)
        self.assertEqual(model.lam, 8)
        self.assertEqual(model.boundary_conditions, 9)


class TestNoOpModel(unittest.TestCase):
    """Tests the NoOpModel object."""

    def setUp(self):
        self.model = NoOpModel(
            sess=None,
            ob_space=Box(low=-1, high=1, shape=(1,)),
            ac_space=Box(low=-2, high=2, shape=(2,)),
            replay_buffer=ReplayBuffer(1, 1, 1, 1),
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