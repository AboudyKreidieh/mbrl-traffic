"""Script containing the LWR model object."""
from mbrl_traffic.models.base import Model


class LWRModel(Model):
    """Lighthill-Whitham-Richards traffic flow model.

    M.J.Lighthill, G.B.Whitham, On kinematic waves II: A theory of traffic flow
    on long, crowded roads. Proceedings of the Royal Society of London Series A
    229, 317-345, 1955

    Attributes
    ----------
    dx : float
        length of individual sections on the highway. Speeds and densities are
        computed on these sections. Must be a factor of the length
    dt : float
        time discretization (in seconds/step)
    v_max : float
        initial speed limit of the LWR model. If not actions are provided
        during the simulation procedure, this value is kept constant throughout
        the simulation
    v_max_max : float
        max speed limit that the network can be assigned
    stream_model : str
        the name of the macroscopic stream model used to denote relationships
        between the current speed and density. Must be one of {"greenshield"}
    boundary_conditions : str
        conditions at road left and right ends; should either dict or string
        ie. {'constant_both': ((density, speed),(density, speed) )}, constant
        value of both ends loop, loop edge values as a ring extend_both,
        extrapolate last value on both ends
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 replay_buffer,
                 verbose,
                 dx,
                 dt,
                 v_max,
                 v_max_max,
                 stream_model,
                 boundary_conditions):
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
        dx : float
            length of individual sections on the highway. Speeds and densities
            are computed on these sections. Must be a factor of the length
        dt : float
            time discretization (in seconds/step)
        v_max : float
            initial speed limit of the LWR model. If not actions are provided
            during the simulation procedure, this value is kept constant
            throughout the simulation
        v_max_max : float
            max speed limit that the network can be assigned
        stream_model : str
            the name of the macroscopic stream model used to denote
            relationships between the current speed and density. Must be one of
            {"greenshield"}
        boundary_conditions : str
            conditions at road left and right ends; should either dict or
            string ie. {'constant_both': ((density, speed),(density, speed) )},
            constant value of both ends loop, loop edge values as a ring
            extend_both, extrapolate last value on both ends
        """
        super(LWRModel, self).__init__(
            sess, ob_space, ac_space, replay_buffer, verbose)

        self.dx = dx
        self.dt = dt
        self.v_max = v_max
        self.v_max_max = v_max_max
        self.stream_model = stream_model
        self.boundary_conditions = boundary_conditions

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
        return {}
