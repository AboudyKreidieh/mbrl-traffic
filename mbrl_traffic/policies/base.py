"""Script containing the base policy object."""


class Policy(object):
    """Base Actor Critic Policy.

    Attributes
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
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 model,
                 replay_buffer,
                 verbose):
        """Instantiate the base policy object.

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
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.model = model
        self.replay_buffer = replay_buffer
        self.verbose = verbose

    def initialize(self):
        """Initialize the policy.

        This is used at the beginning of training by the algorithm, after the
        policy and model parameters have been initialized.
        """
        raise NotImplementedError

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
        tuple < float >
            computed value by the critic
        """
        raise NotImplementedError

    def update(self):
        """Perform a gradient update step.

        Returns
        -------
        float
            actor loss
        tuple < float >
            critic loss
        dict
            additional losses
        """
        raise NotImplementedError

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        raise NotImplementedError

    def save(self, save_path):
        """Save the parameters of a policy.

        Parameters
        ----------
        save_path : str
            Prefix of filenames created for the checkpoint
        """
        raise NotImplementedError

    def load(self, load_path):
        """Load model parameters of a policy.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        raise NotImplementedError
