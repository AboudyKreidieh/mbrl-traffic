"""Script containing the LWR model object."""
import numpy as np

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
    rho_max : float
        maximum density term in the LWR model (in veh/m)
    rho_max_max : float
        maximum possible density of the network (in veh/m)
    v_max : float
        initial speed limit of the LWR model. If not actions are provided
        during the simulation procedure, this value is kept constant throughout
        the simulation
    v_max_max : float
        max speed limit that the network can be assigned
    stream_model : str
        the name of the macroscopic stream model used to denote relationships
        between the current speed and density. Must be one of {"greenshield"}
    lam : float
        exponent of the Green-shield velocity function
    boundary_conditions : str or dict
        conditions at road left and right ends; should either dict or string
        ie. {'constant_both': ((density, speed),(density, speed) )}, constant
        value of both ends loop, loop edge values as a ring extend_both,
        extrapolate last value on both ends
    rho_crit : float
        critical density defined by the Green-shield Model
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 replay_buffer,
                 verbose,
                 dx,
                 dt,
                 rho_max,
                 rho_max_max,
                 v_max,
                 v_max_max,
                 stream_model,
                 lam,
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
        stream_model : str
            the name of the macroscopic stream model used to denote
            relationships between the current speed and density. Must be one of
            {"greenshield"}
        lam : float
            exponent of the Green-shield velocity function
        boundary_conditions : str or dict
            conditions at road left and right ends; should either dict or
            string ie. {'constant_both': ((density, speed),(density, speed) )},
            constant value of both ends loop, loop edge values as a ring
            extend_both, extrapolate last value on both ends
        """
        super(LWRModel, self).__init__(
            sess, ob_space, ac_space, replay_buffer, verbose)

        self.dx = dx
        self.dt = dt
        self.rho_max = rho_max
        self.rho_max_max = rho_max_max
        self.v_max = v_max
        self.v_max_max = v_max_max
        self.stream_model = stream_model
        self.lam = lam
        self.boundary_conditions = boundary_conditions

        # critical density defined by the Green-shield Model
        self.rho_crit = self.rho_max / 2

    def initialize(self):
        """See parent class."""
        pass

    def get_next_obs(self, obs, action):
        """See parent class."""
        del action  # actions are not currently used

        # Advance the density values by one step.
        rho_t = obs[:int(len(obs)/2)]
        rho_tp1 = self._ibvp(rho_t)

        # Compute the new equilibrium speeds.
        v_tp1 = self._v_eq(rho_tp1)

        # Return the new observation.
        return np.concatenate((rho_tp1, v_tp1))

    def update(self):
        """See parent class."""
        pass  # FIXME

    def _ibvp(self, rho_t):
        """Implement Godunov scheme for multi-populations.

        Friedrich, Jan & Kolb, Oliver & Goettlich, Simone. (2018). A Godunov
        type scheme for a class of LWR traffic flow models with non-local flux.
        Networks & Heterogeneous Media. 13. 10.3934/nhm.2018024.

        Parameters
        ----------
        rho_t : array_like
            density data to be analyzed and calculate next points for this data
            using Godunov scheme.
            Note: at time = 0, rho_t = initial density data

        Returns
        -------
        array_like
              next density data points as calculated by the Godunov scheme
        """
        # step = time/distance step
        step = self.dt / self.dx

        # Godunov numerical flux
        f = self._godunov_flux(rho_t)

        fm = np.insert(f[:-1], 0, f[0])

        # Godunov scheme (updating rho_t)
        rho_tp1 = rho_t - step * (f - fm)

        # Update loop boundary conditions for ring-like experiment.
        if self.boundary_conditions == "loop":
            boundary_left = rho_tp1[-1]
            boundary_right = rho_tp1[-2]
            rho_tp1 = np.insert(
                np.append(rho_tp1[1:-1], boundary_right),
                0,
                boundary_left
            )

        # Update boundary conditions by extending/extrapolating boundaries
        # (reduplication).  TODO: is this different from doing nothing?
        if self.boundary_conditions == "extend_both":
            boundary_left = rho_tp1[0]
            boundary_right = rho_tp1[-1]
            rho_tp1 = np.insert(
                np.append(rho_tp1[1:-1], boundary_right),
                0,
                boundary_left
            )

        # Update boundary conditions by keeping boundaries constant.
        if isinstance(self.boundary_conditions, dict):
            if list(self.boundary_conditions.keys())[0] == "constant_both":
                boundary_left = self.boundary_conditions["constant_both"][0]
                boundary_right = self.boundary_conditions["constant_both"][1]
                rho_tp1 = np.insert(
                    np.append(rho_tp1[1:-1], boundary_right),
                    0,
                    boundary_left
                )

        return rho_tp1

    def _godunov_flux(self, rho_t):
        """Calculate the Godunov numerical flux vector of our data.

        Parameters
        ----------
        rho_t : array_like
           densities containing boundary conditions to be analysed

        Returns
        -------
        array_like
            array of fluxes calibrated at every point of our data
        """
        # Collect some relevant variables.
        rho_crit = self.rho_crit
        q_max = rho_crit * self._v_eq(rho_crit)
        v_eq = self._v_eq(rho_t)  # TODO: is this speed or flux?

        # Compute the demand.
        d = rho_t * v_eq * (rho_t < rho_crit) + q_max * (rho_t >= rho_crit)

        # Compute the supply.
        s = rho_t * v_eq * (rho_t > rho_crit) + q_max * (rho_t <= rho_crit)

        # TODO: what is this doing?
        s = np.append(s[1:], s[len(s) - 1])

        # Godunov flux
        return np.minimum(d, s)

    def _v_eq(self, rho_t):
        """Implement the Greenshields model for the equilibrium velocity.

        Greenshields, B. D., Ws Channing, and Hh Miller. "A study of traffic
        capacity." Highway research board proceedings. Vol. 1935. National
        Research Council (USA), Highway Research Board, 1935.

        Parameters
        ----------
        rho_t : array_like
            densities of every specified point on the road

        Returns
        -------
        array_like
            equilibrium velocity at every specified point on road
        """
        return self.v_max * ((1 - (rho_t / self.rho_max)) ** self.lam)

    def get_td_map(self):
        """See parent class."""
        return {}
