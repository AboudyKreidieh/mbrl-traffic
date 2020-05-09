"""Environment for training a variable speed limit."""
import numpy as np
from gym.spaces import Box
from copy import deepcopy
import random

from flow.envs import Env
from flow.core.params import VehicleParams


BASE_ENV_PARAMS = dict(
    # maximum speed limit of segments, in m/s
    max_speed_limit=30,
    # minimum speed limit of segments, in m/s
    min_speed_limit=0
)

RING_ENV_PARAMS = BASE_ENV_PARAMS.copy()
RING_ENV_PARAMS.update(dict(
    # range for the number of vehicles allowed in the network. If set to None,
    # the number of vehicles are is modified from its initial value.
    num_vehicles=[50, 75]
))

MERGE_ENV_PARAMS = BASE_ENV_PARAMS.copy()
MERGE_ENV_PARAMS.update(dict(
    # range for the inflows allowed in the network. If set to None, the inflows
    # are not modified from their initial value.
    inflows=[1000, 2000]
))

HIGHWAY_ENV_PARAMS = BASE_ENV_PARAMS.copy()
HIGHWAY_ENV_PARAMS.update(dict(
    # range for the inflows allowed in the network. If set to None, the inflows
    # are not modified from their initial value.
    inflows=[1000, 2000]
))


class VSLEnv(Env):
    """Environment for training variable speed limit control.

    Required from env_params:
    * max_speed_limit:  maximum speed limit of segments, in m/s
    * min_speed_limit: minimum speed limit of segments, in m/s

    States
        The observation consists of the speeds of the vehicles currently on
        the VSL controller, as well as the speeds of all vehicles in the
        network.

    Actions
        The action space consists of a vector of bounded speed limit for each
        VSL controller $i$.

    Rewards
        The reward provided by the system is equal to the average speed of all
        vehicles in the network.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.

    Attributes
    ----------
    rl_edges : list of str
        the names of the edges controlled by VSL.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in BASE_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(VSLEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

        # FIXME: add to flow?
        # self.rl_edges = deepcopy(self.network.net_params.rl_edges)
        self.rl_edges = deepcopy(self.k.network.get_edge_list())

    @property
    def rl_ids(self):
        """Return the IDs of the RL edges."""
        return self.rl_edges

    @property
    def action_space(self):
        """See class definition."""
        num_rl_edges = len(self.rl_edges)
        return Box(
            low=self.env_params.additional_params['min_speed_limit'],
            high=self.env_params.additional_params['max_speed_limit'],
            shape=(num_rl_edges, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.k.network.set_edge_speed(self.rl_edges, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        reward = np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        return reward

    def get_state(self):
        """See class definition."""
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.k.vehicle.get_ids()]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.k.vehicle.get_ids()]

        return np.array(speed + pos)

    def reset(self):
        """See parent class."""
        return super().reset()


class VSLRingEnv(VSLEnv):
    """A variant of VSL environment for ring network.

    Required from env_params:
    * max_speed_limit:  maximum speed limit of segments, in m/s
    * min_speed_limit: minimum speed limit of segments, in m/s
    * num_vehicles: range for the number of vehicles allowed in the network. If
      set to None, the number of vehicles are is modified from its initial
      value.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in RING_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))
        # this is stored to be reused during the reset procedure
        self._network_cls = network.__class__
        self._network_name = deepcopy(network.orig_name)
        self._network_net_params = deepcopy(network.net_params)
        self._network_initial_config = deepcopy(network.initial_config)
        self._network_traffic_lights = deepcopy(network.traffic_lights)
        self._network_vehicles = deepcopy(network.vehicles)

        super(VSLRingEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    def reset(self):
        """See class definition."""
        # Skip if ring length is None.
        if self.env_params.additional_params["num_vehicles"] is None:
            return super().reset()

        self.step_counter = 1
        self.time_counter = 1

        # Make sure restart instance is set to True when resetting.
        self.sim_params.restart_instance = True

        # Create a new VehicleParams object with a new number of human-
        # driven vehicles.
        n_vehicles = self.env_params.additional_params["num_vehicles"]
        n_vehicles_low = n_vehicles[0]
        n_vehicles_high = n_vehicles[1]
        new_n_vehicles = random.randint(n_vehicles_low, n_vehicles_high)
        params = self._network_vehicles.type_parameters

        print("vehicles: {}".format(new_n_vehicles))

        new_vehicles = VehicleParams()
        new_vehicles.add(
            "human",
            acceleration_controller=params["human"][
                "acceleration_controller"],
            lane_change_controller=params["human"][
                "lane_change_controller"],
            routing_controller=params["human"]["routing_controller"],
            initial_speed=params["human"]["initial_speed"],
            car_following_params=params["human"]["car_following_params"],
            lane_change_params=params["human"]["lane_change_params"],
            num_vehicles=new_n_vehicles)

        # Update the network.
        self.network = self._network_cls(
            self._network_name,
            net_params=self._network_net_params,
            vehicles=new_vehicles,
            initial_config=self._network_initial_config,
            traffic_lights=self._network_traffic_lights,
        )

        return super().reset()


class VSLMergeEnv(VSLEnv):
    """A variant of VSL environment for merge network.

    Required from env_params:
    * max_speed_limit:  maximum speed limit of segments, in m/s
    * min_speed_limit: minimum speed limit of segments, in m/s
    * inflows: range for the inflows allowed in the network. If set to None,
      the inflows are not modified from their initial value.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in MERGE_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(VSLMergeEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    def reset(self):
        """See class definition."""
        # Skip if ring length is None.
        if self.env_params.additional_params["inflows"] is None:
            return super().reset()  # FIXME


class VSLHighwayEnv(VSLEnv):
    """A variant of VSL environment for merge network.

    Required from env_params:
    * max_speed_limit:  maximum speed limit of segments, in m/s
    * min_speed_limit: minimum speed limit of segments, in m/s
    * inflows: range for the inflows allowed in the network. If set to None,
      the inflows are not modified from their initial value.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in HIGHWAY_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(VSLHighwayEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    def reset(self):
        """See class definition."""
        # Skip if ring length is None.
        if self.env_params.additional_params["inflows"] is None:
            return super().reset()  # FIXME
