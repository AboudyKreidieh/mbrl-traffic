"""Example of a merge network with human-driven vehicles.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.
"""
from flow.core.params import SumoParams, EnvParams, NetParams
from flow.core.params import InitialConfig, InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController
from flow.envs.merge import MergePOEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import MergeNetwork

# inflow rate at the highway
FLOW_RATE = 2000

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "a": 0.3,
        "b": 2.0,
        "noise": 0.5,
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0.5,
    ),
    num_vehicles=5)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=FLOW_RATE,
    depart_lane="free",
    depart_speed=30)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=100,
    depart_lane="free",
    depart_speed=20)


flow_params = dict(
    # name of the experiment
    exp_tag='merge-baseline',

    # name of the flow environment the experiment is running on
    env_name=MergePOEnv,

    # name of the network class the experiment is running on
    network=MergeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        use_ballistic=True,
        render=True,
        sim_step=0.5,
        restart_instance=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=7200,
        additional_params=ADDITIONAL_ENV_PARAMS,
        warmup_steps=0,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
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

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        perturbation=5.0,
    ),
)
