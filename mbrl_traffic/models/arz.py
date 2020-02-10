"""Script containing the ARZ model object."""
import tensorflow as tf

from mbrl_traffic.models.base import Model


class ARZModel(Model):
    """Aw–Rascle–Zhang traffic flow model.

    Aw, A. A. T. M., and Michel Rascle. "Resurrection of 'second order' models
    of traffic flow." SIAM journal on applied mathematics 60.3 (2000): 916-938.

    Attributes
    ----------
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 replay_buffer,
                 verbose,
                 dx,
                 rho_max,
                 rho_max_max,
                 v_max,
                 v_max_max,
                 cfl,
                 tau,
                 dt,
                 boundary_conditions,
                 state_dependent,
                 layers,
                 stochastic,
                 num_ensembles,
                 num_particles):
        """Instantiate the ARZ model object.

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
        dx : float
            length of individual sections on the highway. Speeds and densities
            are computed on these sections. Must be a factor of the length
        rho_max : float
            maximum density term in the LWR model (in veh/m)
        rho_max_max : float
            maximum possible density of the network (in veh/m)
        v_max : float
            initial speed limit of the LWR model. If not actions are provided
            during the simulation procedure, this value is kept constant
            throughout the simulation
        v_max_max : float
            max speed limit that the network can be assigned
        cfl : float
            Courant-Friedrichs-Lewy (CFL) condition. Must be a value between 0
            and 1.
        tau : float
            time needed to adjust the velocity of a vehicle from its current
            value to the equilibrium speed (in sec)
        dt : float
            time discretization (in seconds/step)
        boundary_conditions : str
            conditions at road left and right ends; should either dict or
            string ie. {'constant_both': ((density, speed),(density, speed) )},
            constant value of both ends loop, loop edge values as a ring
            extend_both, extrapolate last value on both ends
        state_dependent : bool
            whether the model parameters should be state dependent. In this
            case, a function approximator is learned to map states to model
            parameters.
        layers : list of int
            the size of the neural network for the model, used to compute the
            output of the model parameters for the current step. Used if
            state_dependent is set to True.
        stochastic : bool
            whether the output from the model is stochastic or deterministic
        num_ensembles : int
            number of ensemble models
        num_particles : int
            number of particles used to generate the forward estimate of the
            model.
        """
        super(ARZModel, self).__init__(
            sess, ob_space, ac_space, replay_buffer, verbose)

        self.dx = dx
        self.rho_max = rho_max
        self.rho_max_max = rho_max_max
        self.v_max = v_max
        self.v_max_max = v_max_max
        self.cfl = cfl
        self.tau = tau
        self.dt = dt
        self.boundary_conditions = boundary_conditions
        self.layers = layers
        self.stochastic = stochastic
        self.num_ensembles = num_ensembles
        self.num_particles = num_particles

        # =================================================================== #
        # Part 1. Create the input placeholders.                              #
        # =================================================================== #

        with tf.compat.v1.variable_scope("input", reuse=False):
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.ob_space.shape,
                name='obs_ph')
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.ac_space.shape,
                name='action_ph')
            self.delta_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.ob_space.shape,
                name='delta_ph')

        # =================================================================== #
        # Part 2. Create the outputs from the model.                          #
        # =================================================================== #

        if state_dependent:
            # Create a trainable state-dependent model for each parameter.
            if stochastic:
                params_mean = self._setup_preprocessor(
                    self.obs_ph,
                    self.action_ph,
                    num_params=1,  # FIXME
                    name='params_mean')
                params_logstd = self._setup_preprocessor(
                    self.obs_ph,
                    self.action_ph,
                    num_params=1,  # FIXME
                    name='params_logstd')
                params_std = tf.exp(params_logstd)

                params_out = params_mean + \
                    tf.random.normal(tf.shape(params_mean)) * params_std
            else:
                params_out = self._setup_preprocessor(
                    self.obs_ph,
                    self.action_ph,
                    num_params=1,  # FIXME
                    name='params_out')

        else:
            # Create a single variable for each model parameter.
            if stochastic:
                params_mean = tf.Variable(
                    initial_value=None,  # FIXME
                    shape=(None, None),  # FIXME
                    dtype=tf.float32,
                    name='params_mean')
                params_logstd = tf.Variable(
                    initial_value=None,  # FIXME
                    shape=(None, None),  # FIXME
                    dtype=tf.float32,
                    name='params_logstd')
                params_std = tf.exp(params_logstd)

                params_out = params_mean + \
                    tf.random.normal(tf.shape(params_mean)) * params_std
            else:
                params_out = tf.Variable(
                    initial_value=None,  # FIXME
                    shape=(None, None),  # FIXME
                    dtype=tf.float32,
                    name='params_out')

        self.params = params_out

        # =================================================================== #
        # Part 3. Create the loss and training operation.                     #
        # =================================================================== #

        pass  # TODO

    def _setup_preprocessor(self, obs, action, num_params, name):
        """TODO.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the placeholder for the observations
        action : tf.compat.v1.placeholder
            the placeholder for the actions
        num_params : int
            the number of parameters you are trying to learn
        name : str
            the variable scope

        Returns
        -------
        tf.Variable
            the output (trainable) variable
        """
        with tf.compat.v1.variable_scope(name, reuse=False):
            # Concatenate the input observation and action placeholders.
            pass

            # Create the hidden layers.
            pass

            # Create the output layer.
            pass

        return None

    def initialize(self):
        """See parent class."""
        pass  # FIXME

    def get_next_obs(self, obs, action):
        """See parent class."""
        pass  # FIXME

    def update(self):
        """See parent class."""
        pass  # FIXME

    def get_td_map(self):
        """See parent class."""
        pass  # FIXME
