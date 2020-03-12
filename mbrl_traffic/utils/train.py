"""Utility methods when performing training."""
import argparse
import tensorflow as tf
import gym

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
    rho_max=None,  # FIXME
    # maximum possible density of the network (in veh/m)
    rho_max_max=1,
    # initial speed limit of the model. If not actions are provided during the
    # simulation procedure, this value is kept constant throughout the
    # simulation
    v_max=None,  # FIXME
    # max speed limit that the network can be assigned
    v_max_max=30,
    # the name of the macroscopic stream model used to denote relationships
    # between the current speed and density. Must be one of {"greenshield"}
    stream_model=None,  # FIXME
    # exponent of the Green-shield velocity function
    lam=None,  # FIXME
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
    rho_max=None,  # FIXME
    # maximum possible density of the network (in veh/m)
    rho_max_max=1,
    # initial speed limit of the model. If not actions are provided during the
    # simulation procedure, this value is kept constant throughout the
    # simulation
    v_max=None,  # FIXME
    # max speed limit that the network can be assigned
    v_max_max=30,
    # time needed to adjust the velocity of a vehicle from its current value to
    # the equilibrium speed (in sec)
    tau=None,  # FIXME
    # exponent of the Green-shield velocity function
    lam=None,  # FIXME
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

def parse_params():
    """Parse training options user can specify in command line.

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
        '--env', type=str, help='the name of the environment')

    parser = parse_algorithm_params(parser)
    parser = parse_policy_params(parser)
    parser = parse_model_params(parser)

    return parser


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
        type=str, default="FeedForwardModel",
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
        '--tau',
        type=str, default=ARZ_MODEL_PARAMS["tau"],
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
        '--layer_norm',
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
    # Collect the training and evaluation environments.
    env = create_env(args.env, args.render, evaluate=False)
    eval_env = create_env(args.env, args.render_eval, evaluate=True) \
        if args.evaluate else None

    # Collect the policy and model attributes.
    model_cls, model_params = get_model_params_from_args(args)
    policy_cls, policy_params = get_policy_params_from_args(args)

    return {
        "policy": policy_cls,
        "model": model_cls,
        "env": env,
        "eval_env": eval_env,
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
        "policy_kwargs": policy_params,
        "model_kwargs": model_params,
        "_init_setup_model": True
    }


def create_env(env, render=False, evaluate=False):
    """Return, and potentially create, the environment.

    Parameters
    ----------
    env : str or gym.Env
        the environment, or the name of a registered environment.
    render : bool
        whether to render the environment
    evaluate : bool
        specifies whether this is a training or evaluation environment

    Returns
    -------
    gym.Env or list of gym.Env
        gym-compatible environment(s)
    """
    # Mixed-autonomy traffic environments
    if env.startswith("av"):
        if env == "ring":
            env = None  # TODO
        elif env == "merge":
            env = None  # TODO
        elif env == "highway":
            env = None  # TODO

    # Variable speed limit environments
    elif env.startswith("vsl"):
        if env == "ring":
            env = None  # TODO
        elif env == "merge":
            env = None  # TODO
        elif env == "highway":
            env = None  # TODO

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
            "tau": args.tau,
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
            "layer_norm": args.layer_norm,
            "layers": FEEDFORWARD_MODEL_PARAMS["layers"],
            "act_fun": FEEDFORWARD_MODEL_PARAMS["act_fun"],
            "stochastic": args.stochastic,
            "num_ensembles": args.num_ensembles,
        }

    else:
        raise ValueError("Unknown model: {}".format(args.model))

    return model_cls, model_params
