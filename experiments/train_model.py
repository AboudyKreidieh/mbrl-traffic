"""Script for training a macroscopic or micro-macro dynamics model.

Usage
    python train_model.py "ring-1-lane" --subset "65,70,75"
"""
import argparse
import sys
import tensorflow as tf
import random

from mbrl_traffic.utils.replay_buffer import ReplayBuffer
from mbrl_traffic.utils.train import parse_model_params
from mbrl_traffic.utils.train import get_model_params_from_args
from mbrl_traffic.utils.tf_util import make_session


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a macroscopic or micro-macro dynamics model.",
        epilog="python train_model.py \"ring-1-lane\" --subset \"65,70,75\"")

    # required input parameters
    parser.add_argument(
        'network_type', type=str,
        help='the type of network being used. For example: "ring-1-lane". The '
             'list of possible options can be found in the data folder in the '
             'same location as this script.')

    # optional input parameters
    parser.add_argument(
        '--simulation_type', type=str, default="sumo-idm",
        help='the type of simulation that was used to generate the dataset. '
             'This denotes the type of simulator used as well as the form of '
             'the vehicle acceleration and lane-change dynamics. For example, '
             '"sumo-idm" denotes that the sumo microsimulator was used to '
             'generate the runs, with Flow specifying the IDM model for '
             'car-following dynamics. The possible options can be found in '
             'the <network_type>/baseline folder.')
    parser.add_argument(
        '--subset', type=str, default="",
        help='the subset of the datasets in the provided folder that should '
             'be used, separated by commas. For example, a valid subset when '
             'using the 1-lane ring dataset is "65,70,75". If not specified, '
             'all possible datasets are concatenated and used.')
    parser.add_argument(
        '--steps', type=int, default=1000,
        help='the number of training steps')
    parser.add_argument(
        '--warmup', type=int, default=50,
        help='the number of initial steps from each sub-dataset that are '
             'treated as warmup steps, and subsequently ignored in the '
             'training and testing procedures.')
    parser.add_argument(
        '--training_set', type=float, default=0.8,
        help='the proportion of the dataset to include in the train split. '
             'Should be between 0.0 and 1.0.')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='the training batch size for every step')

    # add model-specific arguments
    parser = parse_model_params(parser)

    return parser.parse_known_args(args)[0]


def import_dataset(network_type, simulation_type, subset, warmup):
    """Import the dataset that the model will be trained on.

    The datasets will consist of a list of (state, action, next_state) tuples,
    where the action is an empty list if no external agent/influence affects
    the dynamics of the network, e.g. in the ring road setting.

    Parameters
    ----------
    network_type : str
        the type of network being used. For example: "ring-1-lane". The list of
        possible options can be found in the data folder in the same location
        as this script.
    simulation_type : str
        the type of simulation that was used to generate the dataset. This
        denotes the type of simulator used as well as the form of the vehicle
        acceleration and lane-change dynamics. For example, "sumo-idm" denotes
        that the sumo microsimulator was used to generate the runs, with Flow
        specifying the IDM model for car-following dynamics. The possible
        options can be found in the <network_type>/baseline folder.
    subset : str
        the subset of the datasets in the provided folder that should be used,
        for example inflow rates in the merge case. If not specified, all
        possible datasets are concatenated and used.
    warmup : int
        the number of initial steps from each sub-dataset that are treated as
        warmup steps, and subsequently ignored in the training and testing
        procedures.

    Returns
    -------
    list of (array_like, array_like, array_like)
        the state, action, next state tuples

    Raises
    ------
    AssertionError
        if the dataset has not been installed. Follow the instructions provided
        alongside this error to install the datasets.
    """
    pass


def create_replay_buffer(args):
    """Create the replay buffer and the testing set.

    Parameters
    ----------
    args : argparse.Namespace
        argument flags from the command line

    Returns
    -------
    mbrl_traffic.utils.replay_buffer.ReplayBuffer
        the replay buffer object with the training dataset already stored
        within it
    list of (array_like, array_like, array_like)
        the testing dataset
    """
    # Import the dataset.
    dataset = import_dataset(
        network_type=args.network_type,
        simulation_type=args.simulation_type,
        subset=args.subset,
        warmup=args.warmup,
    )
    train_size = round(args.training_set * len(dataset))

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Create a replay buffer object.
    replay_buffer = ReplayBuffer(
        buffer_size=train_size,
        batch_size=args.batch_size,
        obs_dim=dataset[0][0].shape,
        ac_dim=dataset[0][1].shape,
    )

    # Store a portion of the data in the replay buffer. Leave the rest for
    # testing.
    for i in range(train_size):
        state, action, next_state = dataset[i]
        replay_buffer.add(
            obs_t=state,
            action=action,
            reward=0,
            obs_tp1=next_state,
            done=False,
        )
    testing_set = dataset[-train_size:]

    return replay_buffer, testing_set


def create_model(args, sess, replay_buffer):
    """Create the specified model object.

    Parameters
    ----------
    args : argparse.Namespace
        argument flags from the command line
    sess : tf.Session
        the tensorflow session
    replay_buffer : mbrl_traffic.utils.replay_buffer.ReplayBuffer
        the replay buffer object

    Returns
    -------
    mbrl_traffic.models.*
        the requested model with the parameters as they were specified in the
        command terminal

    Raises
    ------
    ValueError
        if an unknown model type is specified
    """
    # Some shared variables by all models.
    ob_space = None  # FIXME
    ac_space = None  # FIXME

    model_cls, model_params = get_model_params_from_args(args)

    model = model_cls(
        sess=sess,
        ob_space=ob_space,
        ac_space=ac_space,
        replay_buffer=replay_buffer,
        verbose=2,
        **model_params,
    )

    return model


def main(args):
    """Perform the complete model training operation."""
    # Create a tenfsorflow session.
    graph = tf.Graph()
    with graph.as_default():
        sess = make_session(num_cpu=3, graph=graph)

    # Create the replay buffer and the testing set.
    replay_buffer, testing_set = create_replay_buffer(args)

    # Create the model.
    model = create_model(args, sess, replay_buffer)

    with sess.as_default(), graph.as_default():
        # Perform any model initialization that may be necessary.
        model.initialize()

        for i in range(args.steps):
            # Perform the training procedure.
            pass

            # Evaluate the performance of the model on the training set.
            pass

            # Log the results and store the model parameters.
            pass

    return 0


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])
    main(flags)
