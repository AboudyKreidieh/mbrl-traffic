"""Script containing the NoOP policy object."""
from mbrl_traffic.policies.base import Policy


class NoOpPolicy(Policy):
    """No-Operation policy object.

    Used as a placeholder when purely training the model.
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 model,
                 replay_buffer,
                 verbose):
        """Instantiate the no-op policy object.

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
        """
        super(NoOpPolicy, self).__init__(
            sess, ob_space, ac_space, model, replay_buffer, verbose)

    def initialize(self):
        """Do nothing."""
        pass

    def update(self):
        """Do nothing."""
        return 0, (0,), {}

    def get_action(self, obs, apply_noise, random_actions):
        """Return a random action."""
        return self.ac_space.sample()

    def value(self, obs, action):
        """Return an default value."""
        return 0,

    def get_td_map(self):
        """Return an empty dictionary."""
        return {}

    def save(self, save_path):
        """Do nothing."""
        pass

    def load(self, load_path):
        """Do nothing."""
        pass
