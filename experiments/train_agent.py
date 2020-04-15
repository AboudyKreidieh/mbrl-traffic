"""Script for training automated vehicles in a mixed autonomy network.

Usage

    python train_agent.py "ring" --model "NoOpModel" --policy "SACPolicy"
"""
import os
import json
from time import strftime
import sys
from copy import deepcopy

from mbrl_traffic.algorithm import ModelBasedRLAlgorithm
from mbrl_traffic.utils.misc import ensure_dir
from mbrl_traffic.utils.train import parse_params
from mbrl_traffic.utils.train import get_algorithm_params_from_args
from mbrl_traffic.utils.train import get_model_params_from_args
from mbrl_traffic.utils.train import get_policy_params_from_args


def run_exp(env,
            policy,
            model,
            alg_params,
            policy_params,
            model_params,
            steps,
            dir_name,
            evaluate,
            seed,
            eval_interval,
            log_interval,
            save_interval,
            initial_exploration_steps):
    """Run a single training procedure.

    Parameters
    ----------
    env : str or gym.Env
        the training/testing environment
    policy : type [ mbrl_traffic.policies.base.Policy ]
        the policy class to use
    model : type [ mbrl_traffic.models.base.Model ]
        the model class to use
    alg_params : dict
        additional algorithm hyper-parameters
    policy_params : dict
        policy-specific hyper-parameters
    model_params : dict
        policy-specific hyper-parameters
    steps : int
        total number of training steps
    dir_name : str
        the location the results files are meant to be stored
    evaluate : bool
        whether to include an evaluation environment
    seed : int
        specified the random seed for numpy, tensorflow, and random
    eval_interval : int
        number of simulation steps in the training environment before an
        evaluation is performed
    log_interval : int
        the number of training steps before logging training results
    save_interval : int
        number of simulation steps in the training environment before the model
        is saved
    """
    eval_env = env if evaluate else None

    alg = ModelBasedRLAlgorithm(
        policy=policy,
        model=model,
        env=env,
        eval_env=eval_env,
        policy_kwargs=policy_params,
        model_kwargs=model_params,
        **alg_params
    )

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_dir=dir_name,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_interval=save_interval,
        initial_exploration_steps=initial_exploration_steps,
        seed=seed,
    )


def main(args):
    """Execute multiple training operations."""
    # collect arguments
    args = parse_params(args)

    # Get the algorithm attributes.
    alg_params = get_algorithm_params_from_args(args)

    # Get the policy class and attributes.
    policy, policy_params = get_policy_params_from_args(args)

    # Get the model class and attributes.
    model, model_params = get_model_params_from_args(args)

    # the directory to store log data in
    base_dir = "data/{}-{}".format(policy.__name__, model.__name__)

    for i in range(args.n_training):
        # value of the next seed
        seed = args.seed + i

        # The time when the current experiment started.
        now = strftime("%Y-%m-%d-%H:%M:%S")

        # Create a save directory folder (if it doesn't exist).
        dir_name = os.path.join(base_dir, '{}/{}'.format(args.env_name, now))
        ensure_dir(dir_name)

        # Add the seed for logging purposes.
        params_with_extra = {
            'seed': seed,
            'env_name': args.env_name,
            'policy': "{}".format(policy.__name__),
            'model': "{}".format(model.__name__),
            'date/time': now,
            'alg_params': deepcopy(alg_params),
            'policy_params': deepcopy(policy_params),
            'model_params': deepcopy(model_params),
        }

        # To deal with functions in the parameters to log.
        if 'act_fun' in params_with_extra['policy_params']:
            act_fun = params_with_extra['policy_params']['act_fun']
            params_with_extra['policy_params']['act_fun'] = act_fun.__name__

        # Add the hyperparameters to the folder.
        with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
            json.dump(params_with_extra, f, sort_keys=True, indent=4)

        run_exp(
            env=args.env_name,
            policy=policy,
            model=model,
            alg_params=alg_params,
            model_params=model_params,
            policy_params=policy_params,
            steps=args.total_steps,
            dir_name=dir_name,
            evaluate=args.evaluate,
            seed=seed,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            initial_exploration_steps=args.initial_exploration_steps,
        )


if __name__ == '__main__':
    main(sys.argv[1:])
