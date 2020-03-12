"""Script containing the k-shoot policy object."""
from mbrl_traffic.policies.base import Policy


class KShootPolicy(Policy):
    """K-shoot policy object.

    Attributes
    ----------
    traj_length : int
        trajectory length to compute the sum of returns over
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 model,
                 replay_buffer,
                 verbose,
                 traj_length):
        """Instantiate the K-shoot policy object.

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
        model : mbrl_traffic.models.base.Model
            the model that is being used to simulate the environment
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        replay_buffer : mbrl_traffic.utils.replay_buffer.ReplayBuffer
            the replay buffer object used by the algorithm to store environment
            data
        traj_length : int
            trajectory length to compute the sum of returns over
        """
        super(KShootPolicy, self).__init__(
            sess, ob_space, ac_space, model, replay_buffer, verbose)

        self.traj_length = traj_length

    def initialize(self):
        """Initialize the policy.

        This is used at the beginning of training by the algorithm, after the
        policy and model parameters have been initialized.
        """
        raise NotImplementedError

    def update(self):
        """See parent class.

        No updates are performed by this policy.
        """
        return (0, 0, 0), 0

    def get_action(self, obs, apply_noise, random_actions):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------
        obs : array_like
            the observation
        apply_noise : bool
            whether to add Gaussian noise to the output of the actor. Defaults
            to False
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.

        Returns
        -------
        array_like
            computed action by the policy
        """
        raise NotImplementedError

    def value(self, obs, action):
        """Call the critic methods to compute the value.

        Parameters
        ----------
        obs : array_like
            the observation
        action : array_like
            the actions performed in the given observation

        Returns
        -------
        array_like
            computed value by the critic
        """
        raise NotImplementedError

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        raise NotImplementedError

    def save(self, save_path):
        """See parent class."""
        raise NotImplementedError

    def load(self, load_path):
        """See parent class."""
        raise NotImplementedError
