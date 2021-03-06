"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
from flow.controllers import IDMController
from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.networks import HighwayNetwork
from flow.envs import TestEnv
from flow.networks.highway import ADDITIONAL_NET_PARAMS


# SET UP PARAMETERS FOR THE SIMULATION

# number of steps per rollout
HORIZON = 3600

# inflow rate on the highway in vehicles per hour
HIGHWAY_INFLOW_RATE = 2000
# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 0.0


# SET UP PARAMETERS FOR THE NETWORK

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # length of the highway
    "length": 2500,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 2
})

# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
inflows = InFlows()

# human vehicles
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    car_following_params=SumoCarFollowingParams(
        min_gap=0,
    ),
    acceleration_controller=(IDMController, {
        "a": 0.3,
        "b": 2.0,
        "noise": 0.5,
        "ignore_noise": [(0, 500)],
    }),
)

if PENETRATION_RATE > 0.0:
    vehicles.add(
        "av",
        num_vehicles=0,
        acceleration_controller=(FollowerStopper, {"v_des": 18.0}),
    )

# add human vehicles on the highway
# add human vehicles on the highway
inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(HIGHWAY_INFLOW_RATE * (1 - PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed=15,
    name="idm_highway_inflow")

if PENETRATION_RATE > 0.0:
    inflows.add(
        veh_type="av",
        edge="highway_0",
        vehs_per_hour=int(HIGHWAY_INFLOW_RATE * (PENETRATION_RATE / 100)),
        depart_lane="free",
        depart_speed=15,
        name="av_highway_inflow")

# SET UP FLOW PARAMETERS

flow_params = dict(
    # name of the experiment
    exp_tag='multiagent_highway',

    # name of the flow environment the experiment is running on
    env_name=TestEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=0,
        sims_per_step=1,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        restart_instance=False
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflows,
        additional_params=additional_net_params
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
