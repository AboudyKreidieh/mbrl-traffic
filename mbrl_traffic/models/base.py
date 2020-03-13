"""Script containing the base model object."""


class Model:
    """Base model object.

    The model object is used to estimate next-step observations given current
    state observations and actions. These models may or may not be trainable,
    depending on the choice of input parameters to the given model.

    Attributes
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
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    """

    def __init__(self, sess, ob_space, ac_space, replay_buffer, verbose):
        """Initialize the base model object.

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
        """
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.replay_buffer = replay_buffer
        self.verbose = verbose

    def initialize(self):
        """Initialize the model.

        This is used at the beginning of training by the algorithm, after the
        policy and model parameters have been initialized.
        """
        raise NotImplementedError

    def get_next_obs(self, obs, action):
        """Compute the estimate of the next-step observation by the model.

        Parameters
        ----------
        obs : array_like
            (batch_size, obs_dim) matrix of current step observations
        action : array_like
            (batch_size, ac_dim) matrix of actions by the agent in the current
            state

        Returns
        -------
        array_like
            (batch_size, obs_dim) matrix of next step observations
        """
        raise NotImplementedError

    def update(self):
        """Perform a gradient update step.

        Returns
        -------
        float
            the error associated with the current model
        """
        raise NotImplementedError

    def compute_loss(self, states, actions, next_states):
        """Compute the loss for a given set of data-points.

        This is used for validation purposes.

        Parameters
        ----------
        states : array_like
            (batch_size, obs_dim) matrix of observations
        actions : array_like
            (batch_size, ac_dim) matrix of actions
        next_states : array_like
            (batch_size, obs_dim) matrix of actual next step observations

        Returns
        -------
        float
            the output from the loss function
        """
        raise NotImplementedError

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        raise NotImplementedError

    def save(self, save_path):
        """Save the parameters of a model.

        Parameters
        ----------
        save_path : str
            Prefix of filenames created for the checkpoint
        """
        raise NotImplementedError

    def load(self, load_path):
        """Load model parameters of a model.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        raise NotImplementedError
