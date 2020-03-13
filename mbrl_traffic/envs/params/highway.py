"""Example of vehicles in a section of the I-210."""  # TODO
import os

import flow.config as config
from flow.envs.multiagent.i210 import I210MultiEnv
from flow.envs.multiagent.i210 import ADDITIONAL_ENV_PARAMS
from flow.networks.i210_subnetwork import I210SubNetwork
from flow.networks.i210_subnetwork import EDGES_DISTRIBUTION
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import VehicleParams
from flow.core.params import InFlows
from flow.core.params import SumoLaneChangeParams
from flow.core.params import InitialConfig
from flow.controllers import IDMController


# the path to the network's xml file
NET_TEMPLATE = os.path.join(
    config.PROJECT_PATH,
    "examples/exp_configs/templates/sumo/test2.net.xml")

# create the base vehicle type that will be used for inflows
vehicles = VehicleParams()
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    acceleration_controller=(IDMController, {
        "a": 0.3,
        "b": 2.0,
        "noise": 0.45
    }),
)

inflow = InFlows()
inflow.add(  # main highway
    veh_type="human",
    edge="119257914",
    vehs_per_hour=8378,
    departLane="random",
    departSpeed=23
)


flow_params = dict(
    # name of the experiment
    exp_tag='I-210_subnetwork',

    # name of the flow environment the experiment is running on
    env_name=I210MultiEnv,

    # name of the network class the experiment is running on
    network=I210SubNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # simulation-related parameters
    sim=SumoParams(
        sim_step=0.8,
        render=False,
        color_by_speed=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=4500,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        template=NET_TEMPLATE
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        edges_distribution=EDGES_DISTRIBUTION,
    ),
)
