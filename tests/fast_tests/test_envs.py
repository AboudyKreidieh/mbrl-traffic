"""Tests for the scripts in mbrl_traffic/envs."""
import unittest

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
# from mbrl_traffic.envs.params.highway import flow_params as highway_params


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

    def test_merge(self):  # FIXME
        """Check the validity of the flow_params dict in params/merge.py."""
        inflow = InFlows()
        inflow.add(
            veh_type="human",
            edge="inflow_highway",
            vehs_per_hour=2000,
            depart_lane="free",
            depart_speed=10)
        inflow.add(
            veh_type="human",
            edge="inflow_merge",
            vehs_per_hour=100,
            depart_lane="free",
            depart_speed=7.5)

        expected_flow_params = dict(
            exp_tag='merge-baseline',
            env_name=MergePOEnv,
            network=MergeNetwork,
            simulator='traci',
            sim=SumoParams(
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

    def test_highway(self):  # TODO
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
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_pass(self):
        """TODO."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
