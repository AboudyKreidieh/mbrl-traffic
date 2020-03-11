"""Script containing the fully-connected model object."""
import numpy as np

from mbrl_traffic.models.base import Model


class NoOpModel(Model):
    """No-Operation model object.

    Used as a placeholder when purely training the policy.
    """

    def __init__(self, sess, ob_space, ac_space, replay_buffer, verbose):
        """Instantiate the no-op model object.

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
        super(NoOpModel, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            replay_buffer=replay_buffer,
            verbose=verbose,
        )

        # an empty observation, consisting only of zeros
        self.noop_obs = np.array([0 for _ in range(self.ob_space.shape[0])])

    def initialize(self):
        """Do nothing."""
        pass

    def get_next_obs(self, obs, action):
        """Return an empty observation, consisting only of zeros."""
        return self.noop_obs

    def update(self):
        """Do nothing."""
        return 0

    def compute_loss(self, states, actions, next_states):
        """Return a default value."""
        return 0

    def get_td_map(self):
        """Return an empty dictionary."""
        return {}
