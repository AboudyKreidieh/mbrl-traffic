"""Flow-specific parameters for the multi-lane ring scenario."""
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS

# Number of vehicles in the network
NUM_VEHICLES = 50
# Length of the ring (in meters)
RING_LENGTH = 1500
# Number of lanes in the ring
NUM_LANES = 1

vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {
        "a": 0.3,
        "b": 2.0,
        "noise": 0.5,
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=0.5,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    num_vehicles=NUM_VEHICLES)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["length"] = RING_LENGTH
additional_net_params["lanes"] = NUM_LANES


flow_params = dict(
    # name of the experiment
    exp_tag='multilane-ring',

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        use_ballistic=True,
        render=True,
        sim_step=0.5,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=7200,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="random",
        min_gap=0.5,
    ),
)
