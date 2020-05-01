"""Script for training a macroscopic or micro-macro dynamics model.

Usage
    python train_model.py "ring-1-lane" --subset "65,70,75"
"""
import os
import argparse
import sys
import tensorflow as tf
import random
import csv
import numpy as np
import pandas as pd
from gym.spaces import Box
import time
from time import strftime
from pathlib import Path

from mbrl_traffic.utils.replay_buffer import ReplayBuffer
from mbrl_traffic.utils.train import parse_model_params
from mbrl_traffic.utils.train import get_model_params_from_args
from mbrl_traffic.utils.tf_util import make_session # """FIXME"""


VALID_NETWORK_TYPES = [
    "ring-1-lane",
    "ring-2-lanes",
    "ring-3-lanes",
    "ring-4-lanes",
    "merge-1-lane",
]

VALID_SIMULATION_TYPES = [
    "sumo-idm",
    "sumo-free",
]


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
        '--network_type', type=str, default=VALID_NETWORK_TYPES[0],  # FIXME
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
        '--subset', type=str, default="65",  # FIXME
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
    parser.add_argument(
        '--log_interval', type=int, default=2000,
        help='the number of training steps before logging training results. '
             'Should be between 0 and the number of training steps.')
    parser.add_argument(
        '--save_interval', type=int, default=10000,
        help='the number of training steps before saving model parameters. '
             'Should be between 0 and the number of training steps.')

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
    directory = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "results/{}/baselines/{}".format(network_type, simulation_type))

    # Check that the network and simulation types are valid.
    assert network_type in VALID_NETWORK_TYPES, \
        "Network type '{}' is not known.".format(network_type)
    assert simulation_type in VALID_SIMULATION_TYPES, \
        "Simulation type '{}' is not known.".format(simulation_type)

    # Check that the dataset is available.
    assert os.path.isdir(directory), \
        "Cannot find the simulation data. Please download by running: wget " \
        "https://s3.us-east-2.amazonaws.com/aboudy.experiments/macro/{}/" \
        "baseline/{}.tar.xz".format(network_type, simulation_type)  # FIXME

    # Collect the names of all dataset files.
    postfix = os.listdir(directory) if subset == "" else subset.split(",")
    files = []
    for p in postfix:
        try:
            files.extend([
                os.path.join(directory, "{}/macro/{}".format(p, file)) for file
                in os.listdir(os.path.join(directory, "{}/macro".format(p)))
                if file.endswith(".csv")
            ])
        except NotADirectoryError:  # an image file for example
            pass

    data = []
    for fp in files:
        # Import the next dataset.
        df = pd.read_csv(fp)

        # Separate into states / actions / next_states.
        columns = list(df.columns)
        n_sections = sum(c.startswith("speed") for c in columns)
        v = np.array(df[["speed_{}".format(i) for i in range(n_sections)]])
        d = np.array(df[["density_{}".format(i) for i in range(n_sections)]])
        states = np.concatenate((d, v), axis=1)  # FIXME?

        # Add to the complete dataset (assuming no action for now).
        for i in range(states.shape[0] - warmup - 1):
            data.append((
                states[warmup + i, :],  # states
                states[warmup + i + 1, :],  # next states
                np.array([])  # actions
            ))

    return data


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
    t0 = time.time()
    print("Importing data...")

    # Import the dataset.
    dataset = import_dataset(
        network_type=args.network_type,
        simulation_type=args.simulation_type,
        subset=args.subset,
        warmup=args.warmup,
    )
    train_size = round(args.training_set * len(dataset))
    test_size = len(dataset) - train_size

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Create a replay buffer object.
    replay_buffer = ReplayBuffer(
        buffer_size=train_size,
        batch_size=args.batch_size,
        obs_dim=dataset[0][0].shape[0],
        ac_dim=dataset[0][2].shape[0],
    )

    # Store a portion of the data in the replay buffer.
    for i in range(train_size):
        state, next_state, action = dataset[i]
        replay_buffer.add(
            obs_t=state,
            action=action,
            reward=0,
            obs_tp1=next_state,
            done=False,
        )

    # Leave the rest of the data for testing.
    testing_set = {
        "states": np.array([
            dataset[train_size + i][0] for i in range(test_size)
        ]),
        "next_states": np.array([
            dataset[train_size + i][1] for i in range(test_size)
        ]),
        "actions": np.array([
            dataset[train_size + i][2] for i in range(test_size)
        ]),
    }

    # some verbosity
    print(" Time taken:         %.1f" % (time.time() - t0))
    print(" Replay buffer size: {}".format(len(replay_buffer)))
    print(" Training set size:  {}".format(testing_set["states"].shape[0]))
    print(" Observation shape:  {}".format(replay_buffer.obs_dim))
    print(" Action shape:       {}".format(replay_buffer.ac_dim))
    print("")

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
    ob_space = Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(replay_buffer.obs_dim,),
        dtype=np.float32
    )
    ac_space = Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(replay_buffer.ac_dim,),
        dtype=np.float32
    )

    # Collect the model class and it's parameters.
    model_cls, model_params = get_model_params_from_args(args)

    # Create the model.
    model = model_cls(
        sess=sess,
        ob_space=ob_space,
        ac_space=ac_space,
        replay_buffer=replay_buffer,
        verbose=2,
        **model_params,
    )

    return model


def log_results(log_dir, model, interval, train_loss, test_loss):
    """Record training and testing results.

    Parameters
    ----------
    log_dir : str
        the directory where the statistics should be stored
    model : mbrl_traffic.models.*
        model with the parameters as they were specified in the command terminal
    interval : int
        time step interval to record results
    train_loss : float
        the loss value on train set
    test_loss : float
        the loss value on test set
    """
    # Generic data
    log_data = {
        "interval": interval,
        "train/loss": train_loss,
        "test/loss": test_loss
    }
    #
    # # Data from the td map.
    # td_map = model.get_td_map()

    # Save log_data in a csv file.
    if log_dir is not None:
        file_path = log_dir + "results.csv"
        exists = os.path.exists(file_path)
        with open(file_path, 'a') as f:
            w = csv.DictWriter(f, fieldnames=log_data.keys())
            if not exists:
                w.writeheader()
            w.writerow(log_data)


def main(args):
    """Perform the complete model training operation."""
    # # Create a save directory folder (if it doesn't exist).
    # The time when the current experiment started.
    now = strftime("%Y-%m-%d-%H:%M:%S")
    directory = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "results/{}/{}/{}".format(args.network_type, args.simulation_type, now))
    Path(directory).mkdir(parents=True, exist_ok=True)


    # Create a tensorflow session.
    graph = tf.Graph()
    with graph.as_default():
        sess = make_session(num_cpu=3, graph=graph)

    # Create the replay buffer and the testing set.
    replay_buffer, testing_set = create_replay_buffer(args)

    # # Create the model.
    model = create_model(args, sess, replay_buffer)

    # TODO
    with sess.as_default(), graph.as_default():
        # Perform any model initialization that may be necessary.
        model.initialize()

        for i in range(args.steps):
            # Perform the training procedure.
            train_loss = model.update()

            # Evaluate the performance of the model on the testing set.

            states_ = testing_set["states"]
            next_states_ = testing_set["next_states"]
            actions_ = testing_set["actions"]

            test_loss = model.compute_loss(
                states=states_,
                actions=actions_,
                next_states=next_states_,
            )
    #
            # # Log the results and store the model parameters.
            # if i % args.log_interval == 0:
            #     log_results(directory, model, i, train_loss, test_loss)
            # if i % args.save_interval == 0:
            #     model.save(directory)

    return 0


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])
    main(flags)