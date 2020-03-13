"""Script containing the non-local model object."""
from mbrl_traffic.models.base import Model


class NonLocalModel(Model):
    """Non-local model object."""

    def __init__(self, sess, ob_space, ac_space, replay_buffer, verbose):
        """Instantiate the non-local model object.

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
        super(NonLocalModel, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            replay_buffer=replay_buffer,
            verbose=verbose,
        )

    def initialize(self):
        """See parent class."""
        raise NotImplementedError

    def get_next_obs(self, obs, action):
        """See parent class."""
        raise NotImplementedError

    def update(self):
        """See parent class."""
        raise NotImplementedError

    def compute_loss(self, states, actions, next_states):
        """See parent class."""
        raise NotImplementedError

    def get_td_map(self):
        """See parent class."""
        raise NotImplementedError

    def save(self, save_path):
        """See parent class."""
        raise NotImplementedError

    def load(self, load_path):
        """See parent class."""
        raise NotImplementedError
