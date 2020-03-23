"""Script containing the Soft Actor Critic policy object."""
import tensorflow as tf
import numpy as np
from functools import reduce
import os

from mbrl_traffic.policies.base import Policy
from mbrl_traffic.utils.tf_util import layer
from mbrl_traffic.utils.tf_util import gaussian_likelihood
from mbrl_traffic.utils.tf_util import apply_squashing_func
from mbrl_traffic.utils.tf_util import get_trainable_vars
from mbrl_traffic.utils.tf_util import get_target_updates
from mbrl_traffic.utils.tf_util import reduce_std

# Cap the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACPolicy(Policy):
    """Soft Actor Critic.

    See: https://arxiv.org/pdf/1801.01290.pdf

    Attributes
    ----------
    actor_lr : float
        actor learning rate
    critic_lr : float
        critic learning rate
    tau : float
        target update rate
    gamma : float
        discount factor
    layers : list of int or None
        the size of the Neural network for the policy
    layer_norm : bool
        enable layer normalisation
    act_fun : tf.nn.*
        the activation function to use in the neural network
    use_huber : bool
        specifies whether to use the huber distance function as the loss
        for the critic. If set to False, the mean-squared error metric is
        used instead
    target_entropy : float
        target entropy used when learning the entropy coefficient
    terminals1 : tf.compat.v1.placeholder
        placeholder for the next step terminals
    rew_ph : tf.compat.v1.placeholder
        placeholder for the rewards
    action_ph : tf.compat.v1.placeholder
        placeholder for the actions
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations
    obs1_ph : tf.compat.v1.placeholder
        placeholder for the next step observations
    deterministic_action : tf.Variable
        the output from the deterministic actor
    policy_out : tf.Variable
        the output from the stochastic actor
    logp_pi : tf.Variable
        the log-probability of a given observation given the output action from
        the policy
    logp_action : tf.Variable
        the log-probability of a given observation given a fixed action. Used
        by the hierarchical policy to perform off-policy corrections.
    qf1 : tf.Variable
        the output from the first Q-function
    qf2 : tf.Variable
        the output from the second Q-function
    value_fn : tf.Variable
        the output from the value function
    qf1_pi : tf.Variable
        the output from the first Q-function with the action provided directly
        by the actor policy
    qf2_pi : tf.Variable
        the output from the second Q-function with the action provided directly
        by the actor policy
    log_alpha : tf.Variable
        the log of the entropy coefficient
    alpha : tf.Variable
        the entropy coefficient
    value_target : tf.Variable
        the output from the target value function. Takes as input the next-step
        observations
    target_init_updates : tf.Operation
        an operation that sets the values of the trainable parameters of the
        target actor/critic to match those actual actor/critic
    target_soft_updates : tf.Operation
        soft target update function
    alpha_loss : tf.Operation
        the operation that returns the loss of the entropy term
    alpha_optimizer : tf.Operation
        the operation that updates the trainable parameters of the entropy term
    actor_loss : tf.Operation
        the operation that returns the loss of the actor
    actor_optimizer : tf.Operation
        the operation that updates the trainable parameters of the actor
    critic_loss : tf.Operation
        the operation that returns the loss of the critic
    critic_optimizer : tf.Operation
        the operation that updates the trainable parameters of the critic
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 model,
                 replay_buffer,
                 verbose,
                 actor_lr,
                 critic_lr,
                 tau,
                 gamma,
                 layers,
                 layer_norm,
                 act_fun,
                 use_huber,
                 target_entropy):
        """Instantiate the SAC policy object.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
        model : mbrl_traffic.models.base.Model
            the model that is being used to simulate the environment
        replay_buffer : mbrl_traffic.utils.replay_buffer.ReplayBuffer
            the replay buffer object used by the algorithm to store environment
            data
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        actor_lr : float
            actor learning rate
        critic_lr : float
            critic learning rate
        tau : float
            target update rate
        gamma : float
            discount factor
        layers : list of int or None
            the size of the Neural network for the policy
        layer_norm : bool
            enable layer normalisation
        act_fun : tf.nn.*
            the activation function to use in the neural network
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        target_entropy : float
            target entropy used when learning the entropy coefficient. If set
            to None, a heuristic value is used.
        """
        super(SACPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            model=model,
            replay_buffer=replay_buffer,
            verbose=verbose,
        )

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.layers = layers
        self.layer_norm = layer_norm
        self.act_fun = act_fun
        self.use_huber = use_huber

        if target_entropy is None:
            self.target_entropy = -np.prod(self.ac_space.shape)
        else:
            self.target_entropy = target_entropy

        self._ac_means = 0.5 * (ac_space.high + ac_space.low)
        self._ac_magnitudes = 0.5 * (ac_space.high - ac_space.low)

        # =================================================================== #
        # Step 1: Create input variables.                                     #
        # =================================================================== #

        with tf.compat.v1.variable_scope("input", reuse=False):
            self.terminals1 = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='terminals1')
            self.rew_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='rewards')
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_space.shape,
                name='obs0')
            self.obs1_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_space.shape,
                name='obs1')

        # =================================================================== #
        # Step 2: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.compat.v1.variable_scope("model", reuse=False):
            self.deterministic_action, self.policy_out, self.logp_pi, \
                self.logp_action = self.make_actor(self.obs_ph, self.action_ph)
            self.qf1, self.qf2, self.value_fn = self.make_critic(
                self.obs_ph, self.action_ph,
                create_qf=True, create_vf=True)
            self.qf1_pi, self.qf2_pi, _ = self.make_critic(
                self.obs_ph, self.policy_out,
                create_qf=True, create_vf=False, reuse=True)

            # The entropy coefficient or entropy can be learned automatically,
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            self.log_alpha = tf.compat.v1.get_variable(
                'log_alpha',
                dtype=tf.float32,
                initializer=0.0)
            self.alpha = tf.exp(self.log_alpha)

        with tf.compat.v1.variable_scope("target", reuse=False):
            # Create the value network
            _, _, value_target = self.make_critic(
                self.obs1_ph, create_qf=False, create_vf=True)
            self.value_target = value_target

        # Create the target update operations.
        init, soft = get_target_updates(
            get_trainable_vars('policy/model/value_fns/vf'),
            get_trainable_vars('policy/target/value_fns/vf'),
            tau, verbose)
        self.target_init_updates = init
        self.target_soft_updates = soft

        # =================================================================== #
        # Step 3: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        self._setup_actor_optimizer()
        self._setup_critic_optimizer()

        # =================================================================== #
        # Step 4: Setup the operations for computing model statistics.        #
        # =================================================================== #

        # Setup the running means and standard deviations of the model inputs
        # and outputs.
        self._setup_stats()

        # =================================================================== #
        # Step 5: Create a saver object to store the model parameters.        #
        # =================================================================== #

        self.saver = tf.compat.v1.train.Saver(
            get_trainable_vars("policy/"), max_to_keep=10)

    def make_actor(self, obs, action, reuse=False, scope="pi"):
        """Create the actor variables.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        action : tf.compat.v1.placeholder
            the input action placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the deterministic actor
        tf.Variable
            the output from the stochastic actor
        tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        tf.Variable
            the log-probability of a given observation given a fixed action
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            pi_h = obs

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                pi_h = layer(
                    pi_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # create the output mean
            policy_mean = layer(
                pi_h, self.ac_space.shape[0], 'mean',
                act_fun=None,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

            # create the output log_std
            log_std = layer(
                pi_h, self.ac_space.shape[0], 'log_std',
                act_fun=None,
            )

        # OpenAI Variation to cap the standard deviation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = tf.exp(log_std)

        # Reparameterization trick
        policy = policy_mean + tf.random.normal(tf.shape(policy_mean)) * std
        logp_pi = gaussian_likelihood(policy, policy_mean, log_std)
        logp_ac = gaussian_likelihood(action, policy_mean, log_std)

        # Apply squashing and account for it in the probability
        _, _, logp_ac = apply_squashing_func(
            policy_mean, action, logp_ac)
        deterministic_policy, policy, logp_pi = apply_squashing_func(
            policy_mean, policy, logp_pi)

        return deterministic_policy, policy, logp_pi, logp_ac

    def make_critic(self,
                    obs,
                    action=None,
                    reuse=False,
                    scope="value_fns",
                    create_qf=True,
                    create_vf=True):
        """Create the critic variables.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        action : tf.compat.v1.placeholder
            the input action placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor
        create_qf : bool
            whether to create the Q-functions
        create_vf : bool
            whether to create the value function

        Returns
        -------
        tf.Variable
            the output from the first Q-function. Set to None if `create_qf` is
            False.
        tf.Variable
            the output from the second Q-function. Set to None if `create_qf`
            is False.
        tf.Variable
            the output from the value function. Set to None if `create_vf` is
            False.
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Value function
            if create_vf:
                with tf.compat.v1.variable_scope("vf", reuse=reuse):
                    vf_h = obs

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        vf_h = layer(
                            vf_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    value_fn = layer(
                        vf_h, 1, 'vf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )
            else:
                value_fn = None

            # Double Q values to reduce overestimation
            if create_qf:
                with tf.compat.v1.variable_scope('qf1', reuse=reuse):
                    # concatenate the observations and actions
                    qf1_h = tf.concat([obs, action], axis=-1)

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        qf1_h = layer(
                            qf1_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    qf1 = layer(
                        qf1_h, 1, 'qf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )

                with tf.compat.v1.variable_scope('qf2', reuse=reuse):
                    # concatenate the observations and actions
                    qf2_h = tf.concat([obs, action], axis=-1)

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        qf2_h = layer(
                            qf2_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    qf2 = layer(
                        qf2_h, 1, 'qf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )
            else:
                qf1, qf2 = None, None

        return qf1, qf2, value_fn

    def _setup_critic_optimizer(self):
        """Create minimization operation for critic Q-function.

        Create a `tf.optimizer.minimize` operation for updating critic
        Q-function with gradient descent.

        See Equations (5, 6) in [1], for further information of the Q-function
        update rule.
        """
        if self.verbose >= 2:
            print('setting up critic optimizer')

        scope_name = 'policy/model/value_fns'

        if self.verbose >= 2:
            for name in ['qf1', 'qf2', 'vf']:
                actor_shapes = [
                    var.get_shape().as_list() for var in
                    get_trainable_vars('{}/{}'.format(scope_name, name))]
                actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                       for shape in actor_shapes])
                print('  {} shapes: {}'.format(name, actor_shapes))
                print('  {} params: {}'.format(name, actor_nb_params))

        # Take the min of the two Q-Values (Double-Q Learning)
        min_qf_pi = tf.minimum(self.qf1_pi, self.qf2_pi)

        # Target for Q value regression
        q_backup = tf.stop_gradient(
            self.rew_ph +
            (1 - self.terminals1) * self.gamma * self.value_target)

        # choose the loss function
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        # Compute Q-Function loss
        qf1_loss = loss_fn(q_backup, self.qf1)
        qf2_loss = loss_fn(q_backup, self.qf2)

        # Target for value fn regression
        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.
        v_backup = tf.stop_gradient(min_qf_pi - self.alpha * self.logp_pi)
        value_loss = loss_fn(self.value_fn, v_backup)

        self.critic_loss = (qf1_loss, qf2_loss, value_loss)

        # Combine the loss functions for the optimizer.
        critic_loss = qf1_loss + qf2_loss + value_loss

        # Critic train op
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(self.critic_lr)
        self.critic_optimizer = critic_optimizer.minimize(
            critic_loss,
            var_list=get_trainable_vars(scope_name))

    def _setup_actor_optimizer(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating policy and
        entropy with gradient descent.
        """
        if self.verbose >= 2:
            print('setting up actor and alpha optimizers')

        scope_name = 'policy/model/pi/'

        if self.verbose >= 2:
            actor_shapes = [var.get_shape().as_list()
                            for var in get_trainable_vars(scope_name)]
            actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                   for shape in actor_shapes])
            print('  actor shapes: {}'.format(actor_shapes))
            print('  actor params: {}'.format(actor_nb_params))

        # Take the min of the two Q-Values (Double-Q Learning)
        min_qf_pi = tf.minimum(self.qf1_pi, self.qf2_pi)

        # Compute the entropy temperature loss.
        self.alpha_loss = -tf.reduce_mean(
            self.log_alpha
            * tf.stop_gradient(self.logp_pi + self.target_entropy))

        alpha_optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        self.alpha_optimizer = alpha_optimizer.minimize(
            self.alpha_loss,
            var_list=self.log_alpha)

        # Compute the policy loss
        self.actor_loss = tf.reduce_mean(self.alpha * self.logp_pi - min_qf_pi)

        # Policy train op (has to be separate from value train op, because
        # min_qf_pi appears in policy_loss)
        actor_optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        self.actor_optimizer = actor_optimizer.minimize(
            self.actor_loss,
            var_list=get_trainable_vars(scope_name))

    def _setup_stats(self):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        # rewards
        tf.compat.v1.summary.scalar('rewards', tf.reduce_mean(self.rew_ph))

        # losses
        tf.compat.v1.summary.scalar('alpha_loss', self.alpha_loss)
        tf.compat.v1.summary.scalar('actor_loss', self.actor_loss)
        tf.compat.v1.summary.scalar('Q1_loss', self.critic_loss[0])
        tf.compat.v1.summary.scalar('Q2_loss', self.critic_loss[1])
        tf.compat.v1.summary.scalar('value_loss', self.critic_loss[1])

        tf.compat.v1.summary.scalar(
            'reference_Q1_mean', tf.reduce_mean(self.qf1))
        tf.compat.v1.summary.scalar(
            'reference_Q1_std', reduce_std(self.qf1))

        tf.compat.v1.summary.scalar(
            'reference_Q2_mean', tf.reduce_mean(self.qf2))
        tf.compat.v1.summary.scalar(
            'reference_Q2_std', reduce_std(self.qf2))

        tf.compat.v1.summary.scalar(
            'reference_actor_Q1_mean', tf.reduce_mean(self.qf1_pi))
        tf.compat.v1.summary.scalar(
            'reference_actor_Q1_std', reduce_std(self.qf1_pi))

        tf.compat.v1.summary.scalar(
            'reference_actor_Q2_mean', tf.reduce_mean(self.qf2_pi))
        tf.compat.v1.summary.scalar(
            'reference_actor_Q2_std', reduce_std(self.qf2_pi))

        tf.compat.v1.summary.scalar(
            'reference_action_mean', tf.reduce_mean(self.policy_out))
        tf.compat.v1.summary.scalar(
            'reference_action_std', reduce_std(self.policy_out))

        tf.compat.v1.summary.scalar(
            'reference_log_probability_mean', tf.reduce_mean(self.logp_pi))
        tf.compat.v1.summary.scalar(
            'reference_log_probability_std', reduce_std(self.logp_pi))

    def initialize(self):
        """See parent class."""
        self.sess.run(self.target_init_updates)

    def update(self):  # TODO
        """Perform a gradient update step."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return 0, (0, 0, 0), {"alpha_loss": 0}

        # Get a batch
        obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample()

        # Normalize the actions (bounded between [-1, 1]).
        actions = (actions - self._ac_means) / self._ac_magnitudes

        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        # Collect all update and loss call operations.
        step_ops = [
            self.critic_loss[0],
            self.critic_loss[1],
            self.critic_loss[2],
            self.actor_loss,
            self.alpha_loss,
            self.critic_optimizer,
            self.actor_optimizer,
            self.alpha_optimizer,
            self.target_soft_updates,
        ]

        # Prepare the feed_dict information.
        feed_dict = {
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1
        }

        # Perform the update operations and collect the actor and critic loss.
        q1_loss, q2_loss, vf_loss, actor_loss, alpha_loss, *_ = self.sess.run(
            step_ops, feed_dict)

        return actor_loss, (q1_loss, q2_loss, vf_loss), \
            {"alpha_loss": alpha_loss}

    def get_action(self, obs, apply_noise, random_actions):
        """See parent class."""
        if random_actions:
            return np.array([self.ac_space.sample()])
        elif apply_noise:
            normalized_action = self.sess.run(
                self.policy_out, feed_dict={self.obs_ph: obs})
            return self._ac_magnitudes * normalized_action + self._ac_means
        else:
            normalized_action = self.sess.run(
                self.deterministic_action, feed_dict={self.obs_ph: obs})
            return self._ac_magnitudes * normalized_action + self._ac_means

    def value(self, obs, action):
        """See parent class."""
        # Normalize the actions (bounded between [-1, 1]).
        action = (action - self._ac_means) / self._ac_magnitudes

        return self.sess.run(
            [self.qf1, self.qf2, self.value_fn],
            feed_dict={
                self.obs_ph: obs,
                self.action_ph: action
            }
        )

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return {}

        # Get a batch.
        obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample()

        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        td_map = {
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1
        }

        return td_map

    def save(self, save_path):
        """See parent class."""
        self.saver.save(self.sess, os.path.join(save_path, "policy"))

    def load(self, load_path):
        """See parent class."""
        self.saver.restore(self.sess, load_path)
