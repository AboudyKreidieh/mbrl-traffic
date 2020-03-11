"""TODO."""
import tensorflow as tf

from mbrl_traffic.models import LWRModel
from mbrl_traffic.models import ARZModel
from mbrl_traffic.models import NoOpModel
from mbrl_traffic.models import FeedForwardModel


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
    rho_max_max=None,  # FIXME
    # initial speed limit of the model. If not actions are provided during the
    # simulation procedure, this value is kept constant throughout the
    # simulation
    v_max=None,  # FIXME
    # max speed limit that the network can be assigned
    v_max_max=None,  # FIXME
    # the name of the macroscopic stream model used to denote relationships
    # between the current speed and density. Must be one of {"greenshield"}
    stream_model=None,  # FIXME
    # exponent of the Green-shield velocity function
    lam=None,  # FIXME
    # conditions at road left and right ends; should either dict or string ie.
    # {'constant_both': ((density, speed),(density, speed))}, constant value of
    # both ends loop, loop edge values as a ring extend_both, extrapolate last
    # value on both ends
    boundary_conditions=None,  # FIXME
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
    rho_max_max=None,  # FIXME
    # initial speed limit of the model. If not actions are provided during the
    # simulation procedure, this value is kept constant throughout the
    # simulation
    v_max=None,  # FIXME
    # max speed limit that the network can be assigned
    v_max_max=None,  # FIXME
    # time needed to adjust the velocity of a vehicle from its current value to
    # the equilibrium speed (in sec)
    tau=None,  # FIXME
    # exponent of the Green-shield velocity function
    lam=None,  # FIXME
    # conditions at road left and right ends; should either dict or string ie.
    # {'constant_both': ((density, speed),(density, speed))}, constant value of
    # both ends loop, loop edge values as a ring extend_both, extrapolate last
    # value on both ends
    boundary_conditions=None,  # FIXME
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
#                       Policy parameters for PPOPolicy                       #
# =========================================================================== #

PPO_POLICY_PARAMS = dict(
    # TODO
)


# =========================================================================== #
#                       Policy parameters for SACPolicy                       #
# =========================================================================== #

SAC_POLICY_PARAMS = dict(
    # actor learning rate
    actor_lr=None,
    # critic learning rate
    critic_lr=None,
    # target update rate
    tau=None,
    # discount factor
    gamma=None,
    # the size of the Neural network for the policy
    layers=None,
    # enable layer normalisation
    layer_norm=None,
    # the activation function to use in the neural network
    act_fun=None,
    # specifies whether to use the huber distance function as the loss for the
    # critic. If set to False, the mean-squared error metric is used instead
    use_huber=None,
    # target entropy used when learning the entropy coefficient. If set
    #  to None, a heuristic value is used.
    target_entropy=None,
)


# =========================================================================== #
#                         Command-line parser methods                         #
# =========================================================================== #

def parse_params():
    """

    """
    pass


def parse_algorithm_params():
    """

    """
    pass


def parse_policy_params():
    """

    """
    pass


def parse_model_params(parser):
    """

    """
    # choose the model
    parser.add_argument(
        '--model',
        type=str, default="FeedForwardModel",
        help='the type of model being trained. Must be one of: LWRModel, '
             'ARZModel, NonLocalModel, NoOpModel, or FeedForwardModel')

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
    pass


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
    policy_cls = None
    policy_params = None

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
            "optimizer_cls": args.optimizer_cls,
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
            "optimizer_cls": args.optimizer_cls,
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
