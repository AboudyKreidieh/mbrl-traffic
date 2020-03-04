"""Script algorithm contain the base model-policy RL algorithm class."""
import os
import time
from collections import deque
import csv
import random
from copy import deepcopy
import numpy as np
import tensorflow as tf

from mbrl_traffic.models import ARZModel
from mbrl_traffic.models import FeedForwardModel
from mbrl_traffic.models import LWRModel
from mbrl_traffic.models import NoOpModel
from mbrl_traffic.policies import SACPolicy
from mbrl_traffic.policies import KShootPolicy
from mbrl_traffic.utils.tf_util import make_session
from mbrl_traffic.utils.misc import ensure_dir, create_env
from mbrl_traffic.utils.replay_buffer import ReplayBuffer
from mbrl_traffic.utils.train import LWR_MODEL_PARAMS
from mbrl_traffic.utils.train import ARZ_MODEL_PARAMS
from mbrl_traffic.utils.train import FEEDFORWARD_MODEL_PARAMS
from mbrl_traffic.utils.train import KSHOOT_POLICY_PARAMS
from mbrl_traffic.utils.train import SAC_POLICY_PARAMS


class ModelBasedRLAlgorithm(object):
    """Model-based RL algorithm class.

    Attributes
    ----------
    policy : type [ mbrl_traffic.policies.base.Policy ]
        the policy model to use
    env : gym.Env or str
        the environment to learn from (if registered in Gym, can be str)
    eval_env : gym.Env or str
        the environment to evaluate from (if registered in Gym, can be str)
    nb_eval_episodes : int
        the number of evaluation episodes
    reward_scale : float
        the value the reward should be scaled by
    render : bool
        enable rendering of the training environment
    render_eval : bool
        enable rendering of the evaluation environment
    eval_deterministic : bool
        if set to True, the policy provides deterministic actions to the
        evaluation environment. Otherwise, stochastic or noisy actions are
        returned.
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    action_space : gym.spaces.*
        the action space of the training environment
    observation_space : gym.spaces.*
        the observation space of the training environment
    policy_kwargs : dict
        policy-specific hyperparameters
    graph : tf.Graph
        the current tensorflow graph
    policy_tf : mbrl_traffic.policies.base.Policy
        the policy object
    sess : tf.compat.v1.Session
        the current tensorflow session
    summary : tf.Summary
        tensorboard summary object
    obs : array_like
        the most recent training observation
    episode_step : int
        the number of steps since the most recent rollout began
    episodes : int
        the total number of rollouts performed since training began
    total_steps : int
        the total number of steps that have been executed since training began
    epoch_episode_rewards : list of float
        a list of cumulative rollout rewards from the most recent training
        iterations
    epoch_episode_steps : list of int
        a list of rollout lengths from the most recent training iterations
    epoch_actor_losses : list of float
        the actor loss values from each SGD step in the most recent training
        iteration
    epoch_q1_losses : list of float
        the loss values for the first Q-function from each SGD step in the most
        recent training iteration
    epoch_q2_losses : list of float
        the loss values for the second Q-function from each SGD step in the
        most recent training iteration
    epoch_actions : list of array_like
        a list of the actions that were performed during the most recent
        training iteration
    epoch_q1s : list of float
        a list of the Q1 values that were calculated during the most recent
        training iteration
    epoch_q2s : list of float
        a list of the Q2 values that were calculated during the most recent
    epoch_episodes : int
        the total number of rollouts performed since the most recent training
        iteration began
    epoch : int
        the total number of training iterations
    episode_rewards_history : list of float
        the cumulative return from the last 100 training episodes
    episode_reward : float
        the cumulative reward since the most reward began
    saver : tf.compat.v1.train.Saver
        tensorflow saver object
    trainable_vars : list of str
        the trainable variables
    rew_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last epoch. Used
        for logging purposes.
    rew_history_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last 100
        episodes. Used for logging purposes.
    eval_rew_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation return from the last time
        evaluations occurred. Used for logging purposes.
    eval_success_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation success rate from the last time
        evaluations occurred. Used for logging purposes.
    """

    def __init__(self,
                 policy,
                 model,
                 env,
                 eval_env=None,
                 nb_eval_episodes=50,
                 policy_update_freq=1,
                 model_update_freq=10,
                 buffer_size=200000,
                 batch_size=128,
                 reward_scale=1.,
                 model_reward_scale=1.,
                 render=False,
                 render_eval=False,
                 eval_deterministic=True,
                 verbose=0,
                 policy_kwargs=None,
                 model_kwargs=None,
                 _init_setup_model=True):
        """Instantiate the algorithm object.

        Parameters
        ----------
        policy : type [ mbrl_traffic.policies.base.Policy ]
            the policy to use
        model : type [ mbrl_traffic.models.base.Model ]
            the model to use
        env : gym.Env or str
            the environment to learn from (if registered in Gym, can be str)
        eval_env : gym.Env or str
            the environment to evaluate from (if registered in Gym, can be str)
        nb_eval_episodes : int
            the number of evaluation episodes
        model_update_freq : int
            number of training steps per model update step. This is separate
            from training the policy.
        buffer_size : int
            the max number of transitions to store
        batch_size : int
            the size of the batch for learning the policy
        reward_scale : float
            the value the reward should be scaled by
        model_reward_scale : float
            the value the model reward should be scaled by
        render : bool
            enable rendering of the training environment
        render_eval : bool
            enable rendering of the evaluation environment
        eval_deterministic : bool
            if set to True, the policy provides deterministic actions to the
            evaluation environment. Otherwise, stochastic or noisy actions are
            returned.
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        policy_kwargs : dict
            policy-specific hyperparameters
        model_kwargs : dict
            model-specific hyperparameters
        _init_setup_model : bool
            Whether or not to build the network at the creation of the instance
        """
        self.policy = policy
        self.model = model
        self.env_name = deepcopy(env)
        self.env = create_env(env, render, evaluate=False)
        self.eval_env = create_env(eval_env, render_eval, evaluate=True)
        self.nb_eval_episodes = nb_eval_episodes
        self.policy_update_freq = policy_update_freq
        self.model_update_freq = model_update_freq
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.model_reward_scale = model_reward_scale
        self.render = render
        self.render_eval = render_eval
        self.eval_deterministic = eval_deterministic
        self.verbose = verbose
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.policy_kwargs = {'verbose': verbose}
        self.model_kwargs = {'verbose': verbose}

        # add the default policy kwargs to the policy_kwargs term
        if policy == KShootPolicy:
            self.policy_kwargs.update(KSHOOT_POLICY_PARAMS.copy())
        elif policy == SACPolicy:
            self.policy_kwargs.update(SAC_POLICY_PARAMS.copy())

        # add the default model kwargs to the model_kwargs term
        if model == ARZModel:
            self.model_kwargs.update(ARZ_MODEL_PARAMS.copy())
        elif model == LWRModel:
            self.model_kwargs.update(LWR_MODEL_PARAMS.copy())
        elif model == FeedForwardModel:
            self.model_kwargs.update(FEEDFORWARD_MODEL_PARAMS.copy())

        self.policy_kwargs.update(policy_kwargs or {})
        self.model_kwargs.update(model_kwargs or {})

        # init
        self.graph = None
        self.sess = None
        self.replay_buffer = None
        self.model_tf = None
        self.policy_tf = None
        self.summary = None
        self.obs = None
        self.episode_step = 0
        self.episodes = 0
        self.total_steps = 0
        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.epoch_model_losses = []
        self.epoch_alpha_losses = []
        self.epoch_actor_losses = []
        self.epoch_q1_losses = []
        self.epoch_q2_losses = []
        self.epoch_value_losses = []
        self.epoch_alphas = []
        self.epoch_actions = []
        self.epoch_q1s = []
        self.epoch_q2s = []
        self.epoch_values = []
        self.epoch_episodes = 0
        self.epoch = 0
        self.episode_rewards_history = deque(maxlen=100)
        self.episode_reward = 0
        self.rew_ph = None
        self.rew_history_ph = None
        self.eval_rew_ph = None
        self.eval_success_ph = None
        self.saver = None

        # Create the model variables and operations.
        if _init_setup_model:
            self.trainable_vars = self.setup_model()

    def setup_model(self):
        """Create the graph, session, policy, and summary objects."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create the tensorflow session.
            self.sess = make_session(num_cpu=3, graph=self.graph)

            # Create a replay buffer object.
            self.replay_buffer = ReplayBuffer(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                obs_dim=self.observation_space.shape[0],
                ac_dim=self.action_space.shape[0]
            )

            # Create the model.
            with tf.compat.v1.variable_scope("model", reuse=False):
                self.model_tf = self.model(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    self.replay_buffer,
                    **self.model_kwargs
                )

            # Create the policy.
            with tf.compat.v1.variable_scope("policy", reuse=False):
                self.policy_tf = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    self.model_tf,
                    self.replay_buffer,
                    **self.policy_kwargs
                )

            # for tensorboard logging
            with tf.compat.v1.variable_scope("Train"):
                self.rew_ph = tf.compat.v1.placeholder(tf.float32)
                self.rew_history_ph = tf.compat.v1.placeholder(tf.float32)

            # Add tensorboard scalars for the return, return history, and
            # success rate.
            tf.compat.v1.summary.scalar(
                "Train/return", self.rew_ph)
            tf.compat.v1.summary.scalar(
                "Train/return_history", self.rew_history_ph)

            # Create the tensorboard summary.
            self.summary = tf.compat.v1.summary.merge_all()

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.compat.v1.global_variables_initializer())
                self.policy_tf.initialize()

            return tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    def _policy(self,
                obs,
                apply_noise=True,
                compute_q=True,
                random_actions=False):
        """Get the actions and critic output, from a given observation.

        Parameters
        ----------
        obs : array_like
            the observation
        apply_noise : bool
            enable the noise
        compute_q : bool
            compute the critic output
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.

        Returns
        -------
        list of float
            the action value
        float
            the critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)

        action = self.policy_tf.get_action(
            obs,
            apply_noise=apply_noise,
            random_actions=random_actions
        )

        q_value = self.policy_tf.value(obs, action) if compute_q else None

        return action.flatten(), q_value

    def _store_transition(self,
                          obs0,
                          action,
                          reward,
                          obs1,
                          terminal1,
                          evaluate=False):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : list of float or list of int
            the last observation
        action : list of float or np.ndarray
            the action
        reward : float
            the reward
        obs1 : list fo float or list of int
            the current observation
        terminal1 : bool
            is the episode done
        evaluate : bool
            whether the sample is being provided by the evaluation environment.
            If so, the data is not stored in the replay buffer.
        """
        # Scale the rewards by the provided term.
        reward *= self.reward_scale

        self.policy_tf.store_transition(
            obs0, action, reward, obs1, terminal1, evaluate)

    def learn(self,
              total_timesteps,
              log_dir=None,
              seed=None,
              log_interval=2000,
              eval_interval=50000,
              save_interval=10000,
              initial_exploration_steps=10000):
        """Perform the complete training operation.

        Parameters
        ----------
        total_timesteps : int
            the total number of samples to train on
        log_dir : str
            the directory where the training and evaluation statistics, as well
            as the tensorboard log, should be stored
        seed : int or None
            the initial seed for training, if None: keep current seed
        log_interval : int
            the number of training steps before logging training results
        eval_interval : int
            number of simulation steps in the training environment before an
            evaluation is performed
        save_interval : int
            number of simulation steps in the training environment before the
            model is saved
        initial_exploration_steps : int, optional
            number of timesteps that the policy is run before training to
            initialize the replay buffer with samples
        """
        # Create a saver object.
        self.saver = tf.compat.v1.train.Saver(
            self.trainable_vars,
            max_to_keep=total_timesteps // save_interval
        )

        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)
        ensure_dir(os.path.join(log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        save_path = os.path.join(log_dir, "tb_log")
        writer = tf.compat.v1.summary.FileWriter(save_path)

        # file path for training and evaluation results
        train_filepath = os.path.join(log_dir, "train.csv")
        eval_filepath = os.path.join(log_dir, "eval.csv")

        # Setup the seed value.
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        if self.verbose >= 2:
            print('Using agent with the following configuration:')
            print(str(self.__dict__.items()))

        eval_steps_incr = 0
        save_steps_incr = 0
        start_time = time.time()

        with self.sess.as_default(), self.graph.as_default():
            # Prepare everything.
            self.obs = self.env.reset()

            # Collect preliminary random samples.
            print("Collecting pre-samples...")
            self._collect_samples(run_steps=initial_exploration_steps,
                                  random_actions=True)
            print("Done!")

            # Reset total statistics variables.
            self.episodes = 0
            self.total_steps = 0
            self.episode_rewards_history = deque(maxlen=100)

            while True:
                # Reset epoch-specific variables.
                self.epoch_episodes = 0
                self.epoch_actions = []
                self.epoch_q1s = []
                self.epoch_q2s = []
                self.epoch_values = []
                self.epoch_model_losses = []
                self.epoch_alpha_losses = []
                self.epoch_actor_losses = []
                self.epoch_q1_losses = []
                self.epoch_q2_losses = []
                self.epoch_value_losses = []
                self.epoch_episode_rewards = []
                self.epoch_episode_steps = []

                for _ in range(log_interval):
                    # If the requirement number of time steps has been met,
                    # terminate training.
                    if self.total_steps >= total_timesteps:
                        return

                    # Collect a sample.
                    self._collect_samples()

                    # Train the policy.
                    if self.total_steps % self.policy_update_freq == 0:
                        self._train_policy()

                    # Train the model.
                    if self.total_steps % self.model_update_freq == 0:
                        self._train_model()

                # Log statistics.
                self._log_training(train_filepath, start_time)

                # Evaluate.
                if self.eval_env is not None and \
                        (self.total_steps - eval_steps_incr) >= eval_interval:
                    eval_steps_incr += eval_interval

                    # Run the evaluation operations over the evaluation env(s).
                    # Note that multiple evaluation envs can be provided.
                    if isinstance(self.eval_env, list):
                        eval_rewards = []
                        eval_successes = []
                        eval_info = []
                        for env in self.eval_env:
                            rew, suc, inf = self._evaluate(env)
                            eval_rewards.append(rew)
                            eval_successes.append(suc)
                            eval_info.append(inf)
                    else:
                        eval_rewards, eval_successes, eval_info = \
                            self._evaluate(self.eval_env)

                    # Log the evaluation statistics.
                    self._log_eval(eval_filepath, start_time, eval_rewards,
                                   eval_successes, eval_info)

                # Run and store summary.
                td_map = self.policy_tf.get_td_map()
                # Check if td_map is empty.
                if td_map:
                    td_map.update({
                        self.rew_ph: np.mean(self.epoch_episode_rewards),
                        self.rew_history_ph: np.mean(
                            self.episode_rewards_history),
                    })
                    summary = self.sess.run(self.summary, td_map)
                    writer.add_summary(summary, self.total_steps)

                # Save a checkpoint of the model.
                if (self.total_steps - save_steps_incr) >= save_interval:
                    save_steps_incr += save_interval
                    self.save(os.path.join(log_dir, "checkpoints/itr"))

                # Update the epoch count.
                self.epoch += 1

    def save(self, save_path):
        """Save the parameters of a tensorflow model.

        Parameters
        ----------
        save_path : str
            Prefix of filenames created for the checkpoint
        """
        self.saver.save(self.sess, save_path, global_step=self.total_steps)

    def load(self, load_path):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        self.saver.restore(self.sess, load_path)

    def _collect_samples(self, run_steps=None, random_actions=False):
        """Perform the sample collection operation.

        This method is responsible for executing rollouts for a number of steps
        before training is executed. The data from the rollouts is stored in
        the policy's replay buffer(s).

        Parameters
        ----------
        run_steps : int, optional
            number of steps to collect samples from. If not provided, the value
            defaults to `self.nb_rollout_steps`.
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.
        """
        for _ in range(run_steps or 1):
            # Predict next action. Use random actions when initializing the
            # replay buffer.
            action, (q1_value, q2_value, v_value) = self._policy(
                self.obs,
                apply_noise=True,
                random_actions=random_actions,
                compute_q=True)
            assert action.shape == self.env.action_space.shape

            # Execute next action.
            new_obs, reward, done, info = self.env.step(action)

            # Store a transition in the replay buffer.
            self._store_transition(self.obs, action, reward, new_obs, done)

            # Book-keeping.
            self.total_steps += 1
            self.episode_reward += reward
            self.episode_step += 1
            self.epoch_actions.append(action)
            self.epoch_q1s.append(q1_value)
            self.epoch_q2s.append(q2_value)
            self.epoch_values.append(v_value)

            # Update the current observation.
            self.obs = new_obs.copy()

            if done:
                # Episode done.
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_rewards_history.append(self.episode_reward)
                self.epoch_episode_steps.append(self.episode_step)
                self.episode_reward = 0.
                self.episode_step = 0
                self.epoch_episodes += 1
                self.episodes += 1

                # Reset the environment.
                self.obs = self.env.reset()

    def _train_policy(self):
        """Perform the training operation to the policy.

        Through this method, the actor and critic networks are updated within
        the policy, and the summary information is logged to tensorboard.
        """
        # Run a step of training from batch.
        critic_loss, actor_loss = self.policy_tf.update()

        # Add actor and critic loss information for logging purposes.
        self.epoch_q1_losses.append(critic_loss[0])
        self.epoch_q2_losses.append(critic_loss[1])
        self.epoch_value_losses.append(critic_loss[2])
        self.epoch_alpha_losses.append(actor_loss[0])
        self.epoch_actor_losses.append(actor_loss[1])

    def _train_model(self):
        """Perform the training operation to the model.

        Through this method, the model parameters are optimized and the summary
        information is logged to tensorboard.
        """
        # Run a step of training from batch.
        model_loss = self.model_tf.update()

        # Add the model loss information for logging purposes.
        self.epoch_model_losses.append(model_loss)

    def _evaluate(self, env):
        """Perform the evaluation operation.

        This method runs the evaluation environment for a number of episodes
        and returns the cumulative rewards and successes from each environment.

        Parameters
        ----------
        env : gym.Env
            the evaluation environment that the policy is meant to be tested on

        Returns
        -------
        list of float
            the list of cumulative rewards from every episode in the evaluation
            phase
        list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        dict
            additional information that is meant to be logged
        """
        num_steps = deepcopy(self.total_steps)
        eval_episode_rewards = []
        eval_episode_successes = []
        ret_info = {'initial': [], 'final': [], 'average': []}

        if self.verbose >= 1:
            for _ in range(3):
                print("-------------------")
            print("Running evaluation for {} episodes:".format(
                self.nb_eval_episodes))

        for i in range(self.nb_eval_episodes):
            # Reset the environment.
            eval_obs = env.reset()

            # Reset rollout-specific variables.
            eval_episode_reward = 0.
            eval_episode_step = 0

            while True:
                eval_action, _ = self._policy(
                    eval_obs,
                    apply_noise=not self.eval_deterministic,
                    random_actions=False,
                    compute_q=False)

                obs, eval_r, done, info = env.step(eval_action)

                # FIXME: maybe add a store_eval option
                # Store a transition in the replay buffer. This is just for the
                # purposes of calling features in the store_transition method
                # of the policy.
                # self.replay_buffer.store(
                #     eval_obs, eval_action,
                #     eval_r, obs, False, evaluate=True
                # )

                # Update the previous step observation.
                eval_obs = obs.copy()

                # Increment the reward and step count.
                num_steps += 1
                eval_episode_reward += eval_r
                eval_episode_step += 1

                if done:
                    eval_episode_rewards.append(eval_episode_reward)

                    if self.verbose >= 1:
                        print("%d/%d" % (i + 1, self.nb_eval_episodes))

                    # Exit the loop.
                    break

        if self.verbose >= 1:
            print("Done.")
            print("Average return: {}".format(np.mean(eval_episode_rewards)))
            if len(eval_episode_successes) > 0:
                print("Success rate: {}".format(
                    np.mean(eval_episode_successes)))
            for _ in range(3):
                print("-------------------")
            print("")

        return eval_episode_rewards, eval_episode_successes, ret_info

    def _log_training(self, file_path, start_time):
        """Log training statistics.

        Parameters
        ----------
        file_path : str
            the list of cumulative rewards from every episode in the evaluation
            phase
        start_time : float
            the time when training began. This is used to print the total
            training time.
        """
        # Log statistics.
        duration = time.time() - start_time

        combined_stats = {
            # Rollout statistics.
            'rollout/return': np.mean(self.epoch_episode_rewards),
            'rollout/return_history': np.mean(self.episode_rewards_history),
            'rollout/episode_steps': np.mean(self.epoch_episode_steps),
            'rollout/episodes': self.epoch_episodes,
            'rollout/actions_mean': np.mean(self.epoch_actions),
            'rollout/actions_std': np.std(self.epoch_actions),

            # Total statistics.
            'total/duration': duration,
            'total/steps_per_second': self.total_steps / duration,
            'total/episodes': self.episodes,
            'total/epochs': self.epoch + 1,
            'total/steps': self.total_steps
        }

        # Append model-specific statistics, if needed.
        if self.model != NoOpModel:
            combined_stats.update({
                'train/loss_model': np.mean(self.epoch_model_losses),
            })

        # Append SAC-specific statistics, if needed.
        if self.policy == SACPolicy:
            combined_stats.update({
                # Rollout statistics.
                'rollout/alpha_mean': np.mean(self.epoch_alphas),
                'rollout/Q1_mean': np.mean(self.epoch_q1s),
                'rollout/Q2_mean': np.mean(self.epoch_q2s),
                'rollout/value_mean': np.mean(self.epoch_values),

                # Train statistics.
                'train/loss_alpha': np.mean(self.epoch_alpha_losses),
                'train/loss_actor': np.mean(self.epoch_actor_losses),
                'train/loss_Q1': np.mean(self.epoch_q1_losses),
                'train/loss_Q2': np.mean(self.epoch_q2_losses),
                'train/loss_value': np.mean(self.epoch_value_losses),
            })

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=combined_stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(combined_stats)

        # Print statistics.
        print("-" * 57)
        for key in sorted(combined_stats.keys()):
            val = combined_stats[key]
            print("| {:<25} | {:<25} |".format(key, val))
        print("-" * 57)
        print('')

    def _log_eval(self, file_path, start_time, rewards, successes, info):
        """Log evaluation statistics.

        Parameters
        ----------
        file_path : str
            path to the evaluation csv file
        start_time : float
            the time when training began. This is used to print the total
            training time.
        rewards : array_like
            the list of cumulative rewards from every episode in the evaluation
            phase
        successes : list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        info : dict
            additional information that is meant to be logged
        """
        duration = time.time() - start_time

        if isinstance(info, dict):
            rewards = [rewards]
            successes = [successes]
            info = [info]

        for i, (rew, suc, info_i) in enumerate(zip(rewards, successes, info)):
            if len(suc) > 0:
                success_rate = np.mean(suc)
            else:
                success_rate = 0  # no success rate to log

            evaluation_stats = {
                "duration": duration,
                "total_step": self.total_steps,
                "success_rate": success_rate,
                "average_return": np.mean(rew)
            }
            # Add additional evaluation information.
            evaluation_stats.update(info_i)

            if file_path is not None:
                # Add an evaluation number to the csv file in case of multiple
                # evaluation environments.
                eval_fp = file_path[:-4] + "_{}.csv".format(i)
                exists = os.path.exists(eval_fp)

                # Save evaluation statistics in a csv file.
                with open(eval_fp, "a") as f:
                    w = csv.DictWriter(f, fieldnames=evaluation_stats.keys())
                    if not exists:
                        w.writeheader()
                    w.writerow(evaluation_stats)
