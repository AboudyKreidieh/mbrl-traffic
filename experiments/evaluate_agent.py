"""Script containing the evaluator for pre-trained policies."""
import sys
import argparse
import os
import json
import tensorflow as tf
import time
import numpy as np

from flow.core.util import emission_to_csv

from mbrl_traffic.models import LWRModel
from mbrl_traffic.models import ARZModel
# from mbrl_traffic.models import NonLocalModel
from mbrl_traffic.models import NoOpModel
from mbrl_traffic.models import FeedForwardModel
from mbrl_traffic.policies import KShootPolicy
from mbrl_traffic.policies import SACPolicy
from mbrl_traffic.policies import NoOpPolicy
from mbrl_traffic.utils.misc import ensure_dir
from mbrl_traffic.utils.tf_util import make_session
from mbrl_traffic.utils.train import create_env


FLOW_ENV_NAMES = [
    "av-ring",
    "av-merge",
    "av-highway",
    "vsl-ring",
    "vsl-merge",
    "vsl-highway",
]


def parse_args(args):
    """Parse evaluation options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Simulate and evaluate an agent within an environment.",
        epilog="python evaluate_agent.py <results_dir>")

    # required input parameters
    parser.add_argument(
        'results_dir', type=str,
        help='The location of the logged data during the training procedure. '
             'This will contain the model parameters under the checkpoints '
             'folder.')

    # optional input parameters
    parser.add_argument(
        '--checkpoint_num', type=int, default=None,
        help='the checkpoint number. If set to None, the last checkpoint is '
             'used.')
    parser.add_argument(
        '--runs', type=int, default=1,
        help='the number of simulations to perform')
    parser.add_argument(
        '--no_render', action='store_true',
        help='Specifies whether to run the simulation during runtime.')
    parser.add_argument(
        '--gen_emission', action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation.')

    return parser.parse_known_args(args)[0]


def get_model_cls(model):
    """Get the model class from the name of the model.

    Parameters
    ----------
    model : str
        the name of the model

    Returns
    -------
    type [ mbrl_traffic.models.base.Model ]
        the model class
    """
    if model == "LWRModel":
        return LWRModel
    elif model == "ARZModel":
        return ARZModel
    # elif model == "NonLocalModel":
    #     return NonLocalModel
    elif model == "NoOpModel":
        return NoOpModel
    elif model == "FeedForwardModel":
        return FeedForwardModel
    else:
        raise ValueError("Unknown model: {}".format(model))


def get_policy_cls(policy):
    """Get the policy class from the name of the policy.

    Parameters
    ----------
    policy : str
        the name of the policy

    Returns
    -------
    type [ mbrl_traffic.policies.base.Policy ]
        the policy class
    """
    if policy == "KShootPolicy":
        return KShootPolicy
    elif policy == "SACPolicy":
        return SACPolicy
    elif policy == "NoOpPolicy":
        return NoOpPolicy
    else:
        raise ValueError("Unknown policy: {}".format(policy))


def get_ckpt_num(base_dir, ckpt_num):
    """Get the checkpoint number.

    Parameters
    ----------
    base_dir : str
        the location of the logged data during the training procedure.
    ckpt_num : int
        the checkpoint number. If set to None, the last checkpoint is used.

    Returns
    -------
    int
        the checkpoint number
    """
    if ckpt_num is None:
        # Get the last checkpoint number.
        filenames = os.listdir(os.path.join(base_dir, "checkpoints"))
        metanum = [int(f.split("-")[-1]) for f in filenames]
        ckpt_num = max(metanum)

    return ckpt_num


def get_model_ckpt(base_dir, ckpt_num, model):
    """Get the path to the model checkpoint.

    Parameters
    ----------
    base_dir : str
        the location of the logged data during the training procedure.
    ckpt_num : int
        the checkpoint number
    model : str
        the name of the model

    Returns
    -------
    str
        the path to the checkpoint for the model
    """
    ckpt = os.path.join(base_dir, "checkpoints/itr-{}".format(ckpt_num))

    # Add the file path of the checkpoint.
    if model == "FeedForwardModel":
        ckpt = os.path.join(ckpt, "model")
    elif model == "NoOpModel":
        ckpt = None  # no checkpoint needed
    else:
        ckpt = os.path.join(ckpt, "model.json")

    return ckpt


def get_policy_ckpt(base_dir, ckpt_num, policy):
    """Get the path to the policy checkpoint.

    Parameters
    ----------
    base_dir : str
        the location of the logged data during the training procedure.
    ckpt_num : int
        the checkpoint number
    policy : str
        the name of the policy

    Returns
    -------
    str
        the path to the checkpoint for the policy
    """
    ckpt = os.path.join(base_dir, "checkpoints/itr-{}".format(ckpt_num))

    # Add the file path of the checkpoint.
    if policy == "SACPolicy":
        ckpt = os.path.join(ckpt, "policy")
    elif policy == "NoOpPolicy":
        ckpt = None  # no checkpoint needed
    else:
        ckpt = os.path.join(ckpt, "policy.json")

    return ckpt


def main(args):
    """Perform the evaluation operation over and agent/policy."""
    flags = parse_args(args)

    # Get the data from the hyperparameters.json file.
    with open(os.path.join(flags.results_dir, 'hyperparameters.json')) as f:
        params = json.load(f)
        env_name = params["env_name"]

    # Determine a proper emission_path (for logging purposes).
    if flags.gen_emission:
        emission_path = os.path.join(flags.results_dir, "results")
        ensure_dir(emission_path)
    else:
        emission_path = None

    # Create a tensorflow graph and session.
    graph = tf.Graph()
    with graph.as_default():
        sess = make_session(num_cpu=3, graph=graph)

    with sess.as_default(), graph.as_default():
        # Recreate the environment.
        env = create_env(
            params["env_name"],
            render=not flags.no_render,
            emission_path=emission_path
        )

        horizon = 1000000 if not hasattr(env, "env_params") \
            else env.env_params.horizon

        # Recreate the model.
        with tf.compat.v1.variable_scope("model", reuse=False):
            model_cls = get_model_cls(params["model"])
            model_params = params["model_params"]
            model = model_cls(
                sess=sess,
                ob_space=env.observation_space,
                ac_space=env.action_space,
                replay_buffer=None,
                verbose=2,
                **model_params,
            )

        # Recreate the policy.
        with tf.compat.v1.variable_scope("policy", reuse=False):
            policy_cls = get_policy_cls(params["policy"])
            policy_params = params["policy_params"]
            if policy_params.get("act_fun") == "relu":
                policy_params["act_fun"] = tf.nn.relu
            policy = policy_cls(
                sess=sess,
                ob_space=env.observation_space,
                ac_space=env.action_space,
                model=model,
                replay_buffer=None,
                verbose=2,
                **policy_params,
            )

        # Initialize everything.
        sess.run(tf.compat.v1.global_variables_initializer())
        model.initialize()
        policy.initialize()
        rewards = []

        # Load the pre-trained policy and model parameters.
        ckpt_num = get_ckpt_num(flags.results_dir, flags.checkpoint_num)

        model_ckpt = get_model_ckpt(
            flags.results_dir, ckpt_num, params["model"])
        model.load(model_ckpt)

        policy_ckpt = get_policy_ckpt(
            flags.results_dir, ckpt_num, params["policy"])
        policy.load(policy_ckpt)

        # Run the forward simulation for the given number of runs.
        for i in range(flags.runs):
            obs = env.reset()
            total_reward = 0

            for t in range(horizon):
                action = policy.get_action([obs], False, False)
                obs, reward, done, _ = env.step(action[0])

                # Render the environment.
                if not flags.no_render and env_name not in FLOW_ENV_NAMES:
                    env.render()

                # Some bookkeeping.
                total_reward += reward

                if done:
                    break

            rewards.append(total_reward)
            print("Round {0}, return: {1}".format(i, total_reward))

        print("Average, std return: {}, {}".format(
            np.mean(rewards), np.std(rewards)))

        # Terminate the environment.
        if hasattr(env, "terminate"):
            env.terminate()

        # Store csv data. Note that this is only for Flow environments.
        if flags.gen_emission:
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            # collect the location of the emission file
            emission_filename = "{}-emission.xml".format(env.network.name)
            emission_path = os.path.join(emission_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)

            # Delete the .xml version of the emission file.
            os.remove(emission_path)


if __name__ == "__main__":
    main(sys.argv[1:])
