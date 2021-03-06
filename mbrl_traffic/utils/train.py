"""Utility methods when performing training."""
import argparse
import tensorflow as tf
import gym
from copy import deepcopy

from flow.utils.registry import make_create_env

from mbrl_traffic.policies import KShootPolicy
from mbrl_traffic.policies import SACPolicy
from mbrl_traffic.policies import NoOpPolicy
from mbrl_traffic.models import LWRModel
from mbrl_traffic.models import ARZModel
from mbrl_traffic.models import NoOpModel
from mbrl_traffic.models import FeedForwardModel
from mbrl_traffic.utils.optimizers import NelderMead
from mbrl_traffic.utils.optimizers import GeneticAlgorithm
from mbrl_traffic.utils.optimizers import CrossEntropyMethod
from mbrl_traffic.envs.params.ring import flow_params as ring_params
from mbrl_traffic.envs.params.merge import flow_params as merge_params
from mbrl_traffic.envs.params.highway_multi import flow_params \
    as highway_multi_params
from mbrl_traffic.envs.av import AVOpenEnv
from mbrl_traffic.envs.av import AVClosedEnv
from mbrl_traffic.envs.av import OPEN_ENV_PARAMS
from mbrl_traffic.envs.av import CLOSED_ENV_PARAMS
from mbrl_traffic.envs.vsl import VSLRingEnv
from mbrl_traffic.envs.vsl import VSLMergeEnv
from mbrl_traffic.envs.vsl import VSLHighwayEnv
from mbrl_traffic.envs.vsl import RING_ENV_PARAMS
from mbrl_traffic.envs.vsl import MERGE_ENV_PARAMS
from mbrl_traffic.envs.vsl import HIGHWAY_ENV_PARAMS

# =========================================================================== #
#                        Model parameters for LWRModel                        #
# =========================================================================== #

LWR_MODEL_PARAMS = dict(
    # length of individual sections on the highway. Speeds and densities are
    # computed on these sections. Must be a factor of the length
    dx=50,
    # time discretization (in seconds/step)
    dt=0.5,
    # maximum density term in the model (in veh/m)
    rho_max=1,
    # maximum possible density of the network (in veh/m)
    rho_max_max=1,
    # initial speed limit of the model. If not actions are provided during the
    # simulation procedure, this value is kept constant throughout the
    # simulation
    v_max=30,
    # max speed limit that the network can be assigned
    v_max_max=30,
    # the name of the macroscopic stream model used to denote relationships
    # between the current speed and density. Must be one of {"greenshield"}
    stream_model="greenshield",
    # exponent of the Green-shield velocity function
    lam=1,
    # conditions at road left and right ends; should either dict or string ie.
    # {'constant_both': ((density, speed),(density, speed))}, constant value of
    # both ends loop, loop edge values as a ring extend_both, extrapolate last
    # value on both ends
    boundary_conditions="loop",
    # the optimizer class to use when training the model parameters
    optimizer_cls="GeneticAlgorithm",
)


# =========================================================================== #
#                        Model parameters for ARZModel                        #
# =========================================================================== #

ARZ_MODEL_PARAMS = dict(
    # length of individual sections on the highway. Speeds and densities are
    # computed on these sections. Must be a factor of the length
    dx=50,
    # time discretization (in seconds/step)
    dt=0.5,
    # maximum density term in the model (in veh/m)
    rho_max=1,
    # maximum possible density of the network (in veh/m)
    rho_max_max=1,
    # initial speed limit of the model. If not actions are provided during the
    # simulation procedure, this value is kept constant throughout the
    # simulation
    v_max=30,
    # max speed limit that the network can be assigned
    v_max_max=30,
    # time needed to adjust the velocity of a vehicle from its current value to
    # the equilibrium speed (in sec)
    tau=9,
    # max tau value that can be assigned
    tau_max=25,
    # exponent of the Green-shield velocity function
    lam=1,
    # conditions at road left and right ends; should either dict or string ie.
    # {'constant_both': ((density, speed),(density, speed))}, constant value of
    # both ends loop, loop edge values as a ring extend_both, extrapolate last
    # value on both ends
    boundary_conditions="loop",
    # the optimizer class to use when training the model parameters
    optimizer_cls="GeneticAlgorithm",
)


# =========================================================================== #
#                    Model parameters for FeedForwardModel                    #
# =========================================================================== #

FEEDFORWARD_MODEL_PARAMS = dict(
    # the model learning rate
    model_lr=1e-5,
    # whether to enable layer normalization
    layer_norm=False,
    # the size of the neural network for the model
    layers=[128, 128, 128],
    # the activation function to use in the neural network
    act_fun=tf.nn.relu,
    # whether the output from the model is stochastic or deterministic
    stochastic=True,
    # number of ensemble models
    num_ensembles=1,
)


# =========================================================================== #
#                     Policy parameters for KShootPolicy                      #
# =========================================================================== #

KSHOOT_POLICY_PARAMS = dict(
    # number of particles used to generate the forward estimate of the model.
    num_particles=1,
    # trajectory length to compute the sum of returns over
    traj_length=10,
)


# =========================================================================== #
#                       Policy parameters for SACPolicy                       #
# =========================================================================== #

SAC_POLICY_PARAMS = dict(
    # actor learning rate
    actor_lr=3e-4,
    # critic learning rate
    critic_lr=3e-4,
    # target update rate
    tau=0.005,
    # discount factor
    gamma=0.99,
    # the size of the Neural network for the policy
    layers=[256, 256],
    # enable layer normalisation
    layer_norm=False,
    # the activation function to use in the neural network
    act_fun=tf.nn.relu,
    # specifies whether to use the huber distance function as the loss for the
    # critic. If set to False, the mean-squared error metric is used instead
    use_huber=False,
    # target entropy used when learning the entropy coefficient. If set
    #  to None, a heuristic value is used.
    target_entropy=None,
)


# =========================================================================== #
#                         Command-line parser methods                         #
# =========================================================================== #

def parse_params(args):
    """Parse training options user can specify in command line.

    Parameters
    ----------
    args : list of str
        command-line arguments

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Perform model-based RL training on a traffic network.",
        epilog="python train_agent.py \"ring\" --gamma 0.999")

    parser.add_argument(
        'env_name', type=str, help='the name of the environment')

    # optional input parameters
    parser.add_argument(
        '--n_training', type=int, default=1,
        help='Number of training operations to perform. Each training '
             'operation is performed on a new seed. Defaults to 1.')
    parser.add_argument(
        '--total_steps',  type=int, default=1000000,
        help='Total number of timesteps used during training.')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Sets the seed for numpy, tensorflow, and random.')
    parser.add_argument(
        '--log_interval', type=int, default=2000,
        help='the number of training steps before logging training results')
    parser.add_argument(
        '--eval_interval', type=int, default=50000,
        help='number of simulation steps in the training environment before '
             'an evaluation is performed')
    parser.add_argument(
        '--save_interval', type=int, default=50000,
        help='number of simulation steps in the training environment before '
             'the model is saved')
    parser.add_argument(
        '--initial_exploration_steps', type=int, default=10000,
        help='number of timesteps that the policy is run before training to '
             'initialize the replay buffer with samples')

    parser = parse_algorithm_params(parser)
    parser = parse_policy_params(parser)
    parser = parse_model_params(parser)

    flags, _ = parser.parse_known_args(args)
    return flags


def parse_algorithm_params(parser):
    """Add the algorithm hyperparameters to the parser."""
    parser.add_argument(
        '--evaluate',
        action="store_true",
        help='whether to perform periodic evaluation episodes as well')
    parser.add_argument(
        '--nb_eval_episodes',
        type=int, default=50,
        help='the number of evaluation episodes')
    parser.add_argument(
        '--policy_update_freq',
        type=int, default=1,
        help='number of training steps per policy update step. This is '
             'separate from training the model')
    parser.add_argument(
        '--model_update_freq',
        type=int, default=10,
        help='number of training steps per model update step. This is '
             'separate from training the policy.')
    parser.add_argument(
        '--buffer_size',
        type=int, default=200000,
        help='the max number of transitions to store')
    parser.add_argument(
        '--batch_size',
        type=int, default=128,
        help='the size of the batch for learning the policy')
    parser.add_argument(
        '--reward_scale',
        type=float, default=1.,
        help='the value the reward should be scaled by')
    parser.add_argument(
        '--model_reward_scale',
        type=float, default=1.,
        help='the value the model reward should be scaled by')
    parser.add_argument(
        '--render',
        action="store_true",
        help='enable rendering of the training environment')
    parser.add_argument(
        '--render_eval',
        action="store_true",
        help='enable rendering of the evaluation environment')
    parser.add_argument(
        '--verbose',
        type=int, default=2,
        help='the verbosity level: 0 none, 1 training information, 2 '
             'tensorflow debug')

    return parser


def parse_policy_params(parser):
    """Add the policy hyperparameters to the parser."""
    # choose the policy
    parser.add_argument(
        '--policy',
        type=str, default="SACPolicy",
        help='the type of model being trained. Must be one of: KShootPolicy, '
             'SACPolicy, or NoOpPolicy.')

    # parameters for KShootPolicy
    parser.add_argument(
        '--num_particles',
        type=int, default=KSHOOT_POLICY_PARAMS["num_particles"],
        help='number of particles used to generate the forward estimate of '
             'the model. Used as an input parameter to the KShootPolicy '
             'object.')
    parser.add_argument(
        '--traj_length',
        type=int, default=KSHOOT_POLICY_PARAMS["traj_length"],
        help='trajectory length to compute the sum of returns over. Used as '
             'an input parameter to the KShootPolicy object.')

    # parameters for SACPolicy
    parser.add_argument(
        '--actor_lr',
        type=float, default=SAC_POLICY_PARAMS["actor_lr"],
        help='actor learning rate. Used as an input parameter to the '
             'SACPolicy object.')
    parser.add_argument(
        '--critic_lr',
        type=float, default=SAC_POLICY_PARAMS["critic_lr"],
        help='critic learning rate. Used as an input parameter to the '
             'SACPolicy object.')
    parser.add_argument(
        '--tau',
        type=float, default=SAC_POLICY_PARAMS["tau"],
        help='critic learning rate. Used as an input parameter to the '
             'SACPolicy object.')
    parser.add_argument(
        '--gamma',
        type=float, default=SAC_POLICY_PARAMS["gamma"],
        help='discount factor. Used as an input parameter to the SACPolicy '
             'object.')
    parser.add_argument(
        '--layer_norm',
        action="store_true",
        help='enable layer normalisation. Used as an input parameter to the '
             'SACPolicy object.')
    parser.add_argument(
        '--use_huber',
        action="store_true",
        help='specifies whether to use the huber distance function as the '
             'loss for the critic. If set to False, the mean-squared error '
             'metric is used instead. Used as an input parameter to the '
             'SACPolicy object.')
    parser.add_argument(
        '--target_entropy',
        type=float, default=SAC_POLICY_PARAMS["target_entropy"],
        help='target entropy used when learning the entropy coefficient. If '
             'set to None, a heuristic value is used. Used as an input '
             'parameter to the SACPolicy object.')

    return parser


def parse_model_params(parser):
    """Add the model hyperparameters to the parser."""
    # choose the model
    parser.add_argument(
        '--model',
        type=str, default="NoOpModel",
        help='the type of model being trained. Must be one of: LWRModel, '
             'ARZModel, NonLocalModel, NoOpModel, or FeedForwardModel.')

    # parameters for LWRModel, ARZModel, and NonLocalModel
    parser.add_argument(
        '--optimizer_cls',
        type=str, default=LWR_MODEL_PARAMS["optimizer_cls"],
        help='the optimizer class to use when training the model parameters')

    # parameters for LWRModel and ARZModel
    parser.add_argument(
        '--dx',
        type=float, default=LWR_MODEL_PARAMS["dx"],
        help='length of individual sections on the highway. Speeds and '
             'densities are computed on these sections. Must be a factor of '
             'the length. Used as an input parameter to the LWRModel and '
             'ARZModel objects.')
    parser.add_argument(
        '--dt',
        type=float, default=LWR_MODEL_PARAMS["dt"],
        help='time discretization (in seconds/step). Used as an input '
             'parameter to the LWRModel and ARZModel objects.')
    parser.add_argument(
        '--rho_max',
        type=float, default=LWR_MODEL_PARAMS["rho_max"],
        help='maximum density term in the model (in veh/m). Used as an input '
             'parameter to the LWRModel and ARZModel objects.')
    parser.add_argument(
        '--rho_max_max',
        type=float, default=LWR_MODEL_PARAMS["rho_max_max"],
        help='maximum possible density of the network (in veh/m). Used as an '
             'input parameter to the LWRModel and ARZModel objects.')
    parser.add_argument(
        '--v_max',
        type=float, default=LWR_MODEL_PARAMS["v_max"],
        help='initial speed limit of the model. If not actions are provided '
             'during the simulation procedure, this value is kept constant '
             'throughout the simulation. Used as an input parameter to the '
             'LWRModel and ARZModel objects.')
    parser.add_argument(
        '--v_max_max',
        type=float, default=LWR_MODEL_PARAMS["v_max_max"],
        help='max speed limit that the network can be assigned. Used as an '
             'input parameter to the LWRModel and ARZModel objects.')
    parser.add_argument(
        '--lam',
        type=float, default=LWR_MODEL_PARAMS["lam"],
        help='exponent of the Green-shield velocity function. Used as an '
             'input parameter to the LWRModel and ARZModel objects.')
    parser.add_argument(
        '--boundary_conditions',
        type=str, default=LWR_MODEL_PARAMS["boundary_conditions"],
        help='conditions at road left and right ends; should either dict or '
             'string ie. {\'constant_both\': ((density, speed),(density, '
             'speed))}, constant value of both ends loop, loop edge values as '
             'a ring extend_both, extrapolate last value on both ends. Used '
             'as an input parameter to the LWRModel and ARZModel objects.')

    # parameters for LWRModel
    parser.add_argument(
        '--stream_model',
        type=str, default=LWR_MODEL_PARAMS["stream_model"],
        help='the name of the macroscopic stream model used to denote '
             'relationships between the current speed and density. Must be '
             'one of {"greenshield"}. Used as an input parameter to the '
             'LWRModel object.')

    # parameters for ARZModel
    parser.add_argument(
        '--tau_arz',
        type=float, default=ARZ_MODEL_PARAMS["tau"],
        help='time needed to adjust the velocity of a vehicle from its '
             'current value to the equilibrium speed (in sec). Used as an '
             'input parameter to the ARZModel object.')

    # parameters for FeedForwardModel
    parser.add_argument(
        '--model_lr',
        type=float, default=FEEDFORWARD_MODEL_PARAMS["model_lr"],
        help='the model learning rate. Used as an input parameter to the '
             'FeedForwardModel object.')
    parser.add_argument(
        '--layer_norm_model',
        action="store_true",
        help='whether to enable layer normalization. Used as an input '
             'parameter to the FeedForwardModel object.')
    parser.add_argument(
        '--stochastic',
        action="store_true",
        help='whether the output from the model is stochastic or '
             'deterministic. Used as an input parameter to the '
             'FeedForwardModel object.')
    parser.add_argument(
        '--num_ensembles',
        type=int, default=FEEDFORWARD_MODEL_PARAMS["num_ensembles"],
        help='number of ensemble models. Used as an input parameter to the '
             'FeedForwardModel object.')

    return parser


# =========================================================================== #
#                          Parser processing methods                          #
# =========================================================================== #

def get_algorithm_params_from_args(args):
    """Extract the algorithm features form the command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        argument flags from the command line

    Returns
    -------
    dict
        the parameters for the algorithm as specified in the command line
    """
    return {
        "nb_eval_episodes": args.nb_eval_episodes,
        "policy_update_freq": args.policy_update_freq,
        "model_update_freq": args.model_update_freq,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "reward_scale": args.reward_scale,
        "model_reward_scale": args.model_reward_scale,
        "render": args.render,
        "render_eval": args.render_eval,
        "eval_deterministic": True,  # this is kept fixed currently
        "verbose": args.verbose,
        "_init_setup_model": True
    }


def create_env(env, render=False, evaluate=False, emission_path=None):
    """Return, and potentially create, the environment.

    Parameters
    ----------
    env : str or gym.Env
        the environment, or the name of a registered environment.
    render : bool
        whether to render the environment
    evaluate : bool
        specifies whether this is a training or evaluation environment
    emission_path : str
        path to the folder in which to create the emissions output for Flow
        environments. Emissions output is not generated if this value is not
        specified.

    Returns
    -------
    gym.Env or list of gym.Env
        gym-compatible environment(s)
    """
    # No environment (for evaluation environments):
    if env is None:
        return None

    # Mixed-autonomy traffic environments
    if env.startswith("av"):
        if env.endswith("ring"):
            flow_params = deepcopy(ring_params)
            flow_params["env_name"] = AVClosedEnv
            flow_params["env"].additional_params = deepcopy(CLOSED_ENV_PARAMS)
        elif env.endswith("merge"):
            flow_params = deepcopy(merge_params)
            flow_params["env_name"] = AVOpenEnv
            flow_params["env"].additional_params = deepcopy(OPEN_ENV_PARAMS)
        elif env.endswith("highway-multi"):
            flow_params = deepcopy(highway_multi_params)
            flow_params["env_name"] = AVOpenEnv
            flow_params["env"].additional_params = deepcopy(OPEN_ENV_PARAMS)
        else:
            raise ValueError("Unknown environment type: {}".format(env))

        # Update the render, evaluation, and emission_path attributes.
        flow_params["sim"].render = render
        flow_params["env"].evaluate = evaluate
        flow_params["sim"].emission_path = emission_path

        # Create the environment.
        env_creator, _ = make_create_env(flow_params)
        env = env_creator()

    # Variable speed limit environments
    elif env.startswith("vsl"):
        if env.endswith("ring"):
            flow_params = deepcopy(ring_params)
            flow_params["env_name"] = VSLRingEnv
            flow_params["env"].additional_params = deepcopy(RING_ENV_PARAMS)
        elif env.endswith("merge"):
            flow_params = deepcopy(merge_params)
            flow_params["env_name"] = VSLMergeEnv
            flow_params["env"].additional_params = deepcopy(MERGE_ENV_PARAMS)
        elif env.endswith("highway-multi"):
            flow_params = deepcopy(highway_multi_params)
            flow_params["env_name"] = VSLHighwayEnv
            flow_params["env"].additional_params = deepcopy(HIGHWAY_ENV_PARAMS)
        else:
            raise ValueError("Unknown environment type: {}".format(env))

        # Update the render, evaluation, and emission_path attributes.
        flow_params["sim"].render = render
        flow_params["env"].evaluate = evaluate
        flow_params["sim"].emission_path = emission_path

        # Create the environment.
        env_creator, _ = make_create_env(flow_params)
        env = env_creator()

    # MuJoCo and other gym environments
    elif isinstance(env, str):
        env = gym.make(env)

    # Reset the environment.
    env.reset()

    return env


def get_policy_params_from_args(args):
    """Extract the policy features form the command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        argument flags from the command line

    Returns
    -------
    type [ mbrl_traffic.policies.base.Policy ]
        the policy class to be used
    dict
        the parameters for the policy as specified in the command line

    Raises
    ------
    ValueError
        if an unknown policy type is specified
    """
    if args.policy == "KShootPolicy":
        policy_cls = KShootPolicy
        policy_params = {
            "num_particles": args.num_particles,
            "traj_length": args.traj_length,
        }

    elif args.policy == "SACPolicy":
        policy_cls = SACPolicy
        policy_params = {
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "tau": args.tau,
            "gamma": args.gamma,
            "layers": SAC_POLICY_PARAMS["layers"],  # this is kept fixed
            "layer_norm": args.layer_norm,
            "act_fun": SAC_POLICY_PARAMS["act_fun"],  # this is kept fixed
            "use_huber": args.use_huber,
            "target_entropy": args.target_entropy,
        }

    elif args.policy == "NoOpPolicy":
        policy_cls = NoOpPolicy
        policy_params = {}

    else:
        raise ValueError("Unknown policy: {}".format(args.policy))

    return policy_cls, policy_params


def get_model_params_from_args(args):
    """Extract the model features form the command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        argument flags from the command line

    Returns
    -------
    type [ mbrl_traffic.models.base.Model ]
        the model class to be used
    dict
        the parameters for the model as specified in the command line

    Raises
    ------
    ValueError
        if an unknown model type is specified
    """
    # Get the optimizer class.
    if args.model in ["LWRModel", "ARZModel", "NonLocalModel"]:
        if args.optimizer_cls == "NelderMead":
            optimizer_cls = NelderMead
        elif args.optimizer_cls == "GeneticAlgorithm":
            optimizer_cls = GeneticAlgorithm
        elif args.optimizer_cls == "CrossEntropyMethod":
            optimizer_cls = CrossEntropyMethod
        else:
            raise ValueError("Unknown optimizer:", args.optimizer_cls)
    else:
        optimizer_cls = None

    if args.model == "LWRModel":
        model_cls = LWRModel
        model_params = {
            "dx": args.dx,
            "dt": args.dt,
            "rho_max": args.rho_max,
            "rho_max_max": args.rho_max_max,
            "v_max": args.v_max,
            "v_max_max": args.v_max_max,
            "stream_model": args.stream_model,
            "lam": args.lam,
            "boundary_conditions": args.boundary_conditions,
            "optimizer_cls": optimizer_cls,
        }

    elif args.model == "ARZModel":
        model_cls = ARZModel
        model_params = {
            "dx": args.dx,
            "dt": args.dt,
            "rho_max": args.rho_max,
            "rho_max_max": args.rho_max_max,
            "v_max": args.v_max,
            "v_max_max": args.v_max_max,
            "tau": args.tau_arz,
            "lam": args.lam,
            "boundary_conditions": args.boundary_conditions,
            "optimizer_cls": optimizer_cls,
        }

    elif args.model == "NonLocalModel":
        model_cls = None  # FIXME
        model_params = {

        }  # FIXME

    elif args.model == "NoOpModel":
        model_cls = NoOpModel
        model_params = {}

    elif args.model == "FeedForwardModel":
        model_cls = FeedForwardModel
        model_params = {
            "model_lr": args.model_lr,
            "layer_norm": args.layer_norm_model,
            "layers": FEEDFORWARD_MODEL_PARAMS["layers"],
            "act_fun": FEEDFORWARD_MODEL_PARAMS["act_fun"],
            "stochastic": args.stochastic,
            "num_ensembles": args.num_ensembles,
        }

    else:
        raise ValueError("Unknown model: {}".format(args.model))

    return model_cls, model_params
