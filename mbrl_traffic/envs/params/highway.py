"""TODO."""

flow_params = dict(
    # name of the experiment
    exp_tag='highway',

    # name of the flow environment the experiment is running on
    env_name=None,  # FIXME

    # name of the network class the experiment is running on
    network=None,  # FIXME

    # simulator that is used by the experiment
    simulator='traci',  # FIXME

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=None,  # FIXME

    # environment related parameters (see flow.core.params.EnvParams)
    env=None,  # FIXME

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=None,  # FIXME

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=None,  # FIXME

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=None,  # FIXME
)
