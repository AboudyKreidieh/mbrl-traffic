"""Script containing the fully-connected model object."""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from functools import reduce

from mbrl_traffic.models.base import Model
from mbrl_traffic.utils.tf_util import get_trainable_vars
from mbrl_traffic.utils.tf_util import reduce_std
from mbrl_traffic.utils.tf_util import layer


class FeedForwardModel(Model):
    """Feed-forward network model object.

    Attributes
    ----------
    model_lr : int
        the model learning rate
    layer_norm : bool
        enable layer normalisation
    layers : list of int
        the size of the neural network for the model
    act_fun : tf.nn.*
        the activation function to use in the neural network
    stochastic : bool
        whether the output from the model is stochastic or deterministic
    num_ensembles : int
        number of ensemble models
    model : list of tf.Variable
        the output from each model in the ensemble
    model_loss : list of tf.Variable
        the loss function for each model in the ensemble
    model_optimizer : list of tf.Operation
        the optimizer operation for each model in the ensemble
    obs_ph : list of tf.compat.v1.placeholder
        current step observation placeholder for each model in the ensemble
    obs1_ph : list of tf.compat.v1.placeholder
        next step observation placeholder for each model in the ensemble
    action_ph : list of tf.compat.v1.placeholder
        current step action placeholder for each model in the ensemble
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 replay_buffer,
                 verbose,
                 model_lr,
                 layer_norm,
                 layers,
                 act_fun,
                 stochastic,
                 num_ensembles):
        """Instantiate the fully-connected model object.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
        replay_buffer : mbrl_traffic.utils.replay_buffer.ReplayBuffer
            the replay buffer object used by the algorithm to store environment
            data
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        model_lr : int
            the model learning rate
        layer_norm : bool
            whether to enable layer normalization
        layers : list of int
            the size of the neural network for the model
        act_fun : tf.nn.*
            the activation function to use in the neural network
        stochastic : bool
            whether the output from the model is stochastic or deterministic
        num_ensembles : int
            number of ensemble models
        """
        super(FeedForwardModel, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            replay_buffer=replay_buffer,
            verbose=verbose,
        )

        self.model_lr = model_lr
        self.layer_norm = layer_norm
        self.layers = layers
        self.act_fun = act_fun
        self.stochastic = stochastic
        self.num_ensembles = num_ensembles

        if self.verbose >= 2:
            print('setting up the dynamics model')

        self.model = []
        self.model_loss = []
        self.model_optimizer = []
        self.obs_ph = []
        self.obs1_ph = []
        self.action_ph = []

        # Create clipping terms for the model logstd. See: TODO
        self.max_logstd = tf.Variable(
            np.ones([1, self.ob_space.shape[0]]) / 2.,
            dtype=tf.float32,
            name="max_log_std")
        self.min_logstd = tf.Variable(
            -np.ones([1, self.ob_space.shape[0]]) * 10.,
            dtype=tf.float32,
            name="max_log_std")

        for i in range(self.num_ensembles):
            # Create placeholders for the model.
            self.obs_ph.append(tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.ob_space.shape,
                name="obs0_{}".format(i)))
            self.obs1_ph.append(tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.ob_space.shape,
                name="obs1_{}".format(i)))
            self.action_ph.append(tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.ac_space.shape,
                name="action_{}".format(i)))

            # Create a trainable model of the Worker dynamics.
            model, mean, logstd = self._setup_model(
                obs=self.obs_ph[-1],
                action=self.action_ph[-1],
                ob_space=self.ob_space,
                scope="model_{}".format(i)
            )
            self.model.append(model)

            # The worker model is trained to learn the change in state between
            # two time-steps.
            delta = self.obs1_ph[-1] - self.obs_ph[-1]

            # Computes the log probability of choosing a specific output - used
            # by the loss
            dist = tfp.distributions.MultivariateNormalDiag(
                loc=mean,
                scale_diag=tf.exp(logstd)
            )
            rho_logp = dist.log_prob(delta)

            # Create the model loss.
            model_loss = -tf.reduce_mean(rho_logp)

            # The additional loss term is in accordance with:
            # https://github.com/kchua/handful-of-trials
            model_loss += 0.01 * tf.reduce_sum(self.max_logstd) \
                - 0.01 * tf.reduce_sum(self.min_logstd)

            self.model_loss.append(model_loss)

            # Create an optimizer object.
            optimizer = tf.compat.v1.train.AdamOptimizer(self.model_lr)

            # Create the model optimization technique.
            model_optimizer = optimizer.minimize(
                model_loss,
                var_list=get_trainable_vars('model_{}'.format(i))
            )
            self.model_optimizer.append(model_optimizer)

            # Add the model loss and dynamics to the tensorboard log.
            tf.compat.v1.summary.scalar(
                'model_{}_loss'.format(i), model_loss)
            tf.compat.v1.summary.scalar(
                'model_{}_mean'.format(i), tf.reduce_mean(model))
            tf.compat.v1.summary.scalar(
                'model_{}_std'.format(i), reduce_std(model))

            # Print the shapes of the generated models.
            if self.verbose >= 2:
                scope_name = 'model_{}'.format(i)
                critic_shapes = [var.get_shape().as_list()
                                 for var in get_trainable_vars(scope_name)]
                critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                        for shape in critic_shapes])
                print('  model shapes: {}'.format(critic_shapes))
                print('  model params: {}'.format(critic_nb_params))

    def _setup_model(self, obs, action, ob_space, reuse=False, scope="rho"):
        """Create the trainable parameters of the Worker dynamics model.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the last step observation, not including the context
        action : tf.compat.v1.placeholder
            the action from the Worker policy. May be a function of the
            Manager's trainable parameters
        ob_space : gym.spaces.*
            the observation space, not including the context space
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the Worker dynamics model
        tf.Variable
            the mean of the Worker dynamics model
        tf.Variable
            the log std of the Worker dynamics model
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Concatenate the observations and actions.
            rho_h = tf.concat([obs, action], axis=-1)

            # Create the hidden layers.
            for i, layer_size in enumerate(self.layers):
                rho_h = layer(
                    rho_h, layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # Create the output mean.
            rho_mean = layer(rho_h, ob_space.shape[0], 'model_mean')

            # Create the output logstd term.
            rho_logvar = layer(rho_h, ob_space.shape[0], 'model_logvar')

            # Perform log-std clipping as describe in Appendix A.1 of: TODO
            rho_logvar = self.max_logstd - tf.nn.softplus(
                self.max_logstd - rho_logvar)
            rho_logvar = self.min_logstd + tf.nn.softplus(
                rho_logvar - self.min_logstd)

            rho_logstd = rho_logvar / 2.

            rho_std = tf.exp(rho_logstd)

            # The model samples from its distribution.
            rho = rho_mean + tf.random.normal(tf.shape(rho_mean)) * rho_std

        return rho, rho_mean, rho_logstd

    def initialize(self):
        """See parent class."""
        pass

    def get_next_obs(self, obs, action):
        """See parent class."""
        # Collect the expected delta observation from each model.
        delta_obs = self.sess.run(
            self.model,
            feed_dict={obs_ph: obs for obs_ph in self.obs_ph}
        )

        # Average all deltas together.
        # TODO

        # Return the expected next step observations.
        return obs + delta_obs

    def update(self):
        """See parent class."""
        step_ops = []
        feed_dict = {}

        # Add the step ops and samples for each of the model training
        # operations in the ensemble.
        for i in range(self.num_ensembles):
            # Sample new data for this model.
            obs, action, reward, obs1, done = self.replay_buffer.sample()

            # Add the training operation.
            step_ops.append(self.model_optimizer[i])

            # Add the samples to the feed_dict.
            feed_dict.update({
                self.obs_ph[i]: obs,
                self.obs1_ph[i]: obs1,
                self.action_ph[i]: action
            })

        # Run the training operations.
        vals = self.sess.run(step_ops, feed_dict=feed_dict)

        # Return the mean model loss.
        return np.mean(vals[:round(len(vals)/2)])

    def get_td_map(self):
        """See parent class."""
        return {}  # FIXME
