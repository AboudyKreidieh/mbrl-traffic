"""Tests for the scripts in mbrl_traffic/envs."""
import unittest
import numpy as np
import random
from copy import deepcopy

from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.envs import AccelEnv
from flow.envs import MergePOEnv
from flow.networks import RingNetwork
from flow.networks import MergeNetwork

from mbrl_traffic.envs.params.ring import flow_params as ring_params
from mbrl_traffic.envs.params.merge import flow_params as merge_params
from mbrl_traffic.envs.av import AVEnv
from mbrl_traffic.envs.av import AVClosedEnv
from mbrl_traffic.envs.av import AVOpenEnv
from mbrl_traffic.envs.av import CLOSED_ENV_PARAMS
from mbrl_traffic.envs.av import OPEN_ENV_PARAMS


class TestParams(unittest.TestCase):
    """Tests the Model object."""

    def test_ring(self):
        """Check the validity of the flow_params dict in params/ring.py."""
        expected_flow_params = dict(
            exp_tag='multilane-ring',
            env_name=AccelEnv,
            network=RingNetwork,
            simulator='traci',
            sim=SumoParams(
                use_ballistic=True,
                render=True,
                sim_step=0.5,
            ),
            env=EnvParams(
                horizon=7200,
                additional_params={
                    'max_accel': 3,
                    'max_decel': 3,
                    'target_velocity': 10,
                    'sort_vehicles': False,
                },
            ),
            net=NetParams(
                additional_params={
                    "length": 1500,
                    "lanes": 1,
                    "speed_limit": 30,
                    "resolution": 40,
                },
            ),
            initial=InitialConfig(
                spacing="random",
                min_gap=0.5,
                shuffle=True,
            ),
        )

        actual_flow_params = ring_params.copy()

        # check the inflows
        self.assertEqual(actual_flow_params["net"].inflows.__dict__,
                         expected_flow_params["net"].inflows.__dict__)
        del actual_flow_params["net"].inflows
        del expected_flow_params["net"].inflows

        # check that each of the parameter match
        for param in ["env", "sim", "net", "initial"]:
            self.assertEqual(actual_flow_params[param].__dict__,
                             expected_flow_params[param].__dict__)

        for param in ["env_name", "network"]:
            self.assertEqual(actual_flow_params[param].__name__,
                             expected_flow_params[param].__name__)

    def test_merge(self):
        """Check the validity of the flow_params dict in params/merge.py."""
        inflow = InFlows()
        inflow.add(
            veh_type="human",
            edge="inflow_highway",
            vehs_per_hour=2000,
            depart_lane="free",
            depart_speed=30)
        inflow.add(
            veh_type="human",
            edge="inflow_merge",
            vehs_per_hour=100,
            depart_lane="free",
            depart_speed=20)

        expected_flow_params = dict(
            exp_tag='merge-baseline',
            env_name=MergePOEnv,
            network=MergeNetwork,
            simulator='traci',
            sim=SumoParams(
                use_ballistic=True,
                render=True,
                sim_step=0.5,
                restart_instance=False,
            ),
            env=EnvParams(
                horizon=7200,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "target_velocity": 25,
                    "num_rl": 5,
                },
                warmup_steps=0,
            ),
            net=NetParams(
                inflows=inflow,
                additional_params={
                    "merge_length": 100,
                    "pre_merge_length": 1100,
                    "post_merge_length": 300,
                    "merge_lanes": 1,
                    "highway_lanes": 1,
                    "speed_limit": 30,
                },
            ),
            initial=InitialConfig(
                spacing="uniform",
                perturbation=5.0,
            ),
        )

        actual_flow_params = merge_params.copy()

        # check the inflows
        self.assertEqual(actual_flow_params["net"].inflows.__dict__,
                         expected_flow_params["net"].inflows.__dict__)
        del actual_flow_params["net"].inflows
        del expected_flow_params["net"].inflows

        # check that each of the parameter match
        for param in ["env", "sim", "net", "initial"]:
            self.assertEqual(actual_flow_params[param].__dict__,
                             expected_flow_params[param].__dict__)

        for param in ["env_name", "network"]:
            self.assertEqual(actual_flow_params[param].__name__,
                             expected_flow_params[param].__name__)

    def test_highway(self):
        """Check the validity of the flow_params dict in params/highway.py."""
        pass  # TODO


class TestVSL(unittest.TestCase):
    """Tests the variable-speed limit environment."""

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


class TestAV(unittest.TestCase):
    """Tests the automated vehicles environment."""

    def setUp(self):
        self.sim_params = deepcopy(ring_params)["sim"]
        self.sim_params.render = False

        # for AVClosedEnv
        flow_params_closed = deepcopy(ring_params)

        self.network_closed = flow_params_closed["network"](
            name="test_closed",
            vehicles=flow_params_closed["veh"],
            net_params=flow_params_closed["net"],
        )
        self.env_params_closed = flow_params_closed["env"]
        self.env_params_closed.additional_params = CLOSED_ENV_PARAMS.copy()

        # for AVOpenEnv
        flow_params_open = deepcopy(merge_params)

        self.network_open = flow_params_open["network"](
            name="test_closed",
            vehicles=flow_params_open["veh"],
            net_params=flow_params_open["net"],
        )
        self.env_params_open = flow_params_open["env"]
        self.env_params_open.additional_params = OPEN_ENV_PARAMS.copy()

    def test_base_env(self):
        """Validate the functionality of the AVEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the observation space matches its expected values
           a. for the single lane case
           b. for the multi-lane case
        3. that the action space matches its expected values
           a. for the single lane case
           b. for the multi-lane case
        4. that the observed vehicle IDs after a reset matches its expected
           values
           a. for the single lane case
           b. for the multi-lane case
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVEnv,
                sim_params=self.sim_params,
                network=self.network_open,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "penalty": 1,
                },
            )
        )

        # Set a random seed.
        random.seed(0)
        np.random.seed(0)

        # Create a single lane environment.
        env_single = AVEnv(
            env_params=self.env_params_closed,
            sim_params=self.sim_params,
            network=self.network_closed
        )

        # Create a multi-lane environment.
        env_multi = None  # TODO
        del env_multi  # TODO: remove

        # test case 2.a
        self.assertTrue(
            test_space(
                gym_space=env_single.observation_space,
                expected_size=5 * env_single.initial_vehicles.num_rl_vehicles,
                expected_min=-float("inf"),
                expected_max=float("inf"),
            )
        )

        # test case 2.b
        pass  # TODO

        # test case 3.a
        self.assertTrue(
            test_space(
                gym_space=env_single.action_space,
                expected_size=env_single.initial_vehicles.num_rl_vehicles,
                expected_min=-1,
                expected_max=1,
            )
        )

        # test case 3.b
        pass  # TODO

        # test case 4.a
        self.assertTrue(
            test_observed(
                env_class=AVEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                env_params=self.env_params_closed,
                expected_observed=['rl_1', 'rl_2', 'rl_3', 'rl_4', 'human_0',
                                   'human_44', 'rl_0']
            )
        )

        # test case 4.b
        pass  # TODO

    def test_closed_env(self):
        """Validate the functionality of the AVClosedEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2, that the number of vehicles is properly modified in between resets
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVClosedEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "penalty": 1,
                    "num_vehicles": [50, 75],
                    "sort_vehicles": True,
                },
            )
        )

        # set a random seed to ensure the network lengths are always the same
        # during testing
        random.seed(1)

        # test case 2
        env = AVClosedEnv(
            env_params=self.env_params_closed,
            sim_params=self.sim_params,
            network=self.network_closed
        )

        # reset the network several times and check its length
        self.assertEqual(env.k.vehicle.num_vehicles, 50)
        self.assertEqual(env.k.vehicle.num_rl_vehicles, 5)
        env.reset()
        self.assertEqual(env.k.vehicle.num_vehicles, 54)
        self.assertEqual(env.k.vehicle.num_rl_vehicles, 5)
        env.reset()
        self.assertEqual(env.k.vehicle.num_vehicles, 75)
        self.assertEqual(env.k.vehicle.num_rl_vehicles, 5)

    def test_open_env(self):
        """Validate the functionality of the AVOpenEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2, that the inflow rate of vehicles is properly modified in between
           resets
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVOpenEnv,
                sim_params=self.sim_params,
                network=self.network_open,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "penalty": 1,
                    "inflows": [1000, 2000],
                    "rl_penetration": 0.1,
                    "num_rl": 5,
                },
            )
        )

        # test case 2
        pass  # TODO


###############################################################################
#                              Utility methods                                #
###############################################################################

def test_additional_params(env_class,
                           sim_params,
                           network,
                           additional_params):
    """Test that the environment raises an Error in any param is missing.

    Parameters
    ----------
    env_class : flow.envs.Env type
        blank
    sim_params : flow.core.params.SumoParams
        sumo-specific parameters
    network : flow.networks.Network
        network that works for the environment
    additional_params : dict
        the valid and required additional parameters for the environment in
        EnvParams

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    for key in additional_params.keys():
        # remove one param from the additional_params dict
        new_add = additional_params.copy()
        del new_add[key]

        try:
            env_class(
                sim_params=sim_params,
                network=network,
                env_params=EnvParams(additional_params=new_add)
            )
            # if no KeyError is raised, the test has failed, so return False
            return False
        except KeyError:
            # if a KeyError is raised, test the next param
            pass

    # make sure that add all params does not lead to an error
    try:
        env_class(
            sim_params=sim_params,
            network=network,
            env_params=EnvParams(additional_params=additional_params.copy())
        )
    except KeyError:
        # if a KeyError is raised, the test has failed, so return False
        return False

    # if removing all additional params led to KeyErrors, the test has passed,
    # so return True
    return True


def test_space(gym_space, expected_size, expected_min, expected_max):
    """Test that an action or observation space is the correct size and bounds.

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

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    return gym_space.shape[0] == expected_size \
        and all(gym_space.high == expected_max) \
        and all(gym_space.low == expected_min)


def test_observed(env_class,
                  sim_params,
                  network,
                  env_params,
                  expected_observed):
    """Test that the observed vehicles in the environment are as expected.

    Parameters
    ----------
    env_class : flow.envs.Env class
        blank
    sim_params : flow.core.params.SumoParams
        sumo-specific parameters
    network : flow.networks.Network
        network that works for the environment
    env_params : flow.core.params.EnvParams
        environment-specific parameters
    expected_observed : array_like
        expected list of observed vehicles

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    env = env_class(sim_params=sim_params,
                    network=network,
                    env_params=env_params)
    env.reset()
    env.step(env.action_space.sample())
    env.additional_command()
    test_mask = np.all(
        np.array(env.k.vehicle.get_observed_ids()) ==
        np.array(expected_observed)
    )
    env.terminate()

    return test_mask


if __name__ == '__main__':
    unittest.main()
