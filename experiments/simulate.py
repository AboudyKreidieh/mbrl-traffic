"""Runner script for non-RL simulations in flow.

Usage
    python simulate.py EXP_CONFIG --no_render
"""
import argparse
import sys
from copy import deepcopy

from flow.core.experiment import Experiment
from flow.core.params import AimsunParams

from mbrl_traffic.envs.params.ring import flow_params as ring_params
from mbrl_traffic.envs.params.merge import flow_params as merge_params
from mbrl_traffic.envs.params.highway_multi import flow_params \
    as highway_multi_params
from mbrl_traffic.envs.params.highway_single import flow_params \
    as highway_single_params
from mbrl_traffic.envs.params.ring import RING_LENGTH, NUM_LANES


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python simulate.py EXP_CONFIG --num_runs INT --no_render")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/non_rl.')

    # optional input parameters
    parser.add_argument(
        '--num_runs', type=int, default=1,
        help='Number of simulations to run. Defaults to 1.')
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Specifies whether to run the simulation during runtime.')
    parser.add_argument(
        '--aimsun',
        action='store_true',
        help='Specifies whether to run the simulation using the simulator '
             'Aimsun. If not specified, the simulator used is SUMO.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation.')

    # new to this repository
    parser.add_argument(
        '--ring_length', type=float, default=RING_LENGTH,
        help='length of the ring if using the ring network')
    parser.add_argument(
        '--ring_lanes', type=float, default=NUM_LANES,
        help='the number of lanes if using the the ring network')
    # TODO: anything else?

    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])
    env_name = flags.exp_config

    # Get the flow_params object.
    if env_name == "ring":
        flow_params = deepcopy(ring_params)
        flow_params['net'].additional_params['length'] = flags.ring_length
        flow_params['net'].additional_params['lanes'] = flags.ring_lanes
    elif env_name == "merge":
        flow_params = deepcopy(merge_params)
    elif env_name == "highway-single":
        flow_params = deepcopy(highway_single_params)
    elif env_name == "highway-multi":
        flow_params = deepcopy(highway_multi_params)
    else:
        raise ValueError("exp_config must be one of 'ring' or 'highway'")

    # Update some variables based on inputs.
    flow_params['sim'].render = not flags.no_render
    flow_params['simulator'] = 'aimsun' if flags.aimsun else 'traci'

    # Update sim_params if using Aimsun.
    if flags.aimsun:
        sim_params = AimsunParams()
        sim_params.__dict__.update(flow_params['sim'].__dict__)
        flow_params['sim'] = sim_params

    # specify an emission path if they are meant to be generated
    if flags.gen_emission:
        flow_params['sim'].emission_path = "./data"

    # Create the experiment object.
    exp = Experiment(flow_params)

    # Run for the specified number of rollouts.
    exp.run(flags.num_runs, convert_to_csv=flags.gen_emission)
