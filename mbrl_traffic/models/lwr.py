"""Script containing the LWR model object."""
import numpy as np
import os
import json
from copy import deepcopy
from sklearn.metrics import mean_squared_error

from mbrl_traffic.models.base import Model

# maximum value for the lambda term in the Greenshield model
LAM_MAX = 4


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
        speed limit of the LWR model
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
    optimizer_cls : type [ mbrl_traffic.utils.optimizers.base.Optimizer ]
        the optimizer class to use when training the model parameters
    optimizer_params : dict
        optimizer-specific parameters
    optimizer : mbrl_traffic.utils.optimizers.base.Optimizer
        the optimizer object
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
                 boundary_conditions,
                 optimizer_cls,
                 optimizer_params=None):
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
            initial speed limit of the model. If not actions are provided
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
        optimizer_cls : type [ mbrl_traffic.utils.optimizers.base.Optimizer ]
            the optimizer class to use when training the model parameters
        optimizer_params : dict
            optimizer-specific parameters
        """
        super(LWRModel, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            replay_buffer=replay_buffer,
            verbose=verbose,
        )

        self.dx = dx
        self.dt = dt
        self.rho_max = rho_max
        self.rho_max_max = rho_max_max
        self.v_max = v_max
        self.v_max_max = v_max_max
        self.stream_model = stream_model
        self.lam = lam
        self.boundary_conditions = boundary_conditions
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params or {}

        def fitness_fn(x):
            """Return the loss used by the optimizer.

            This method temporarily updates the model parameters in the class
            and computes the loss for a batch of data in the replay buffer.

            Parameters
            ----------
            x : list of float
                the model parameters, of the form: [rho_max, lam]

            Returns
            -------
            float
                the loss associated with the model parameters
            """
            # Store the original model parameters.
            orig_rho_max = deepcopy(self.rho_max)
            orig_v_max = deepcopy(self.v_max)
            orig_lam = deepcopy(self.lam)

            # Update the model parameter based on the input.
            self.rho_max = x[0]
            self.v_max = x[1]
            self.lam = x[2]

            # Sample a batch of state/action/next_state from the replay buffer.
            states, actions, next_states, _, _ = self.replay_buffer.sample()

            # Compute the loss.
            loss = self.compute_loss(states, actions, next_states)

            # Set the model parameters back to their original values.
            self.rho_max = orig_rho_max
            self.v_max = orig_v_max
            self.lam = orig_lam

            return loss

        # Create the optimizer object.
        self.optimizer = optimizer_cls(
            param_low=[0, 0, 0],
            param_high=[rho_max_max, v_max_max, LAM_MAX],
            fitness_fn=fitness_fn,
            verbose=verbose,
            **self.optimizer_params
        )

    def initialize(self):
        """See parent class."""
        pass

    def get_next_obs(self, obs, action):
        """See parent class."""
        del action  # actions are not currently used

        # Advance the density values by one step.
        rho_t = obs[:int(len(obs)/2)]  # FIXME: is this right?
        rho_tp1 = self._ibvp(rho_t)

        # Compute the new equilibrium speeds.
        v_tp1 = self._v_eq(rho_tp1)

        # Return the new observation.
        return np.concatenate((rho_tp1, v_tp1))

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
        # (reduplication).
        elif self.boundary_conditions == "extend_both":
            pass

        # Update boundary conditions by keeping boundaries constant.
        elif isinstance(self.boundary_conditions, dict) and \
                list(self.boundary_conditions.keys())[0] == "constant_both":
            boundary_left = self.boundary_conditions["constant_both"][0]
            boundary_right = self.boundary_conditions["constant_both"][1]
            rho_tp1 = np.insert(
                np.append(rho_tp1[1:-1], boundary_right),
                0,
                boundary_left
            )

        else:
            raise ValueError("Unknown boundary conditions: {}".format(
                self.boundary_conditions))

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
        # critical density defined by the Green-shield Model
        rho_crit = self.rho_max / 2

        v_eq = self._v_eq(rho_t)
        q_max = rho_crit * v_eq

        # Compute the demand.
        d = rho_t * v_eq * (rho_t < rho_crit) + q_max * (rho_t >= rho_crit)

        # Compute the supply.
        s = rho_t * v_eq * (rho_t > rho_crit) + q_max * (rho_t <= rho_crit)

        # TODO: what is this doing?
        s = np.append(s[1:], s[-1])

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

    def update(self):
        """See parent class."""
        model_params, error = self.optimizer.solve()

        # Update the model parameters in the class.
        self.rho_max = model_params[0]
        self.v_max = model_params[1]
        self.lam = model_params[2]

        return error

    def compute_loss(self, states, actions, next_states):
        """See parent class."""
        expected_next_states = []
        for i in range(states.shape[0]):
            # Compute the predicted next states.
            expected_next_states.append(
                self.get_next_obs(states[i, :], actions[i, :]))
        expected_next_states = np.asarray(expected_next_states)

        sections = round(next_states.shape[1] / 2)
        actual_y = next_states[:, :sections]
        expected_y = expected_next_states[:, :sections]

        # Compute the loss.
        return mean_squared_error(actual_y, expected_y)

    def get_td_map(self):
        """See parent class."""
        return {}

    def save(self, save_path):
        """See parent class."""
        params = {
            "dx": self.dx,
            "dt": self.dt,
            "rho_max": self.rho_max,
            "rho_max_max": self.rho_max_max,
            "v_max": self.v_max,
            "v_max_max": self.v_max_max,
            "stream_model": self.stream_model,
            "lam": self.lam,
            "boundary_conditions": self.boundary_conditions,
        }

        with open(os.path.join(save_path, "model.json"), 'w') as f:
            json.dump(params, f, sort_keys=True, indent=4)

    def load(self, load_path):
        """See parent class."""
        with open(load_path) as f:
            data = json.load(f)

        self.dx = data["dx"]
        self.dt = data["dt"]
        self.rho_max = data["rho_max"]
        self.rho_max_max = data["rho_max_max"]
        self.v_max = data["v_max"]
        self.v_max_max = data["v_max_max"]
        self.stream_model = data["stream_model"]
        self.lam = data["lam"]
        self.boundary_conditions = data["boundary_conditions"]
