"""Script containing the LWR model object."""
from mbrl_traffic.models.base import Model


class LWRModel(Model):
    """Lighthill-Whitham-Richards traffic flow model.

    M.J.Lighthill, G.B.Whitham, On kinematic waves II: A theory of traffic flow
    on long, crowded roads. Proceedings of the Royal Society of London Series A
    229, 317-345, 1955

    Attributes
    ----------
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 replay_buffer,
                 verbose):
        """Instantiate the LWR model object.

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
        super(LWRModel, self).__init__(
            sess, ob_space, ac_space, replay_buffer, verbose)

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
