"""Script containing the ARZ model object."""
import numpy as np
from scipy.optimize import fsolve
import os
import json
from copy import deepcopy
from sklearn.metrics import mean_squared_error

from mbrl_traffic.models.base import Model

# maximum value for the lambda term in the Greenshield model
LAM_MAX = 4


class ARZModel(Model):
    """Aw–Rascle–Zhang traffic flow model.

    Aw, A. A. T. M., and Michel Rascle. "Resurrection of 'second order' models
    of traffic flow." SIAM journal on applied mathematics 60.3 (2000): 916-938.

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
    tau : float
        time needed to adjust the velocity of a vehicle from its current value
        to the equilibrium speed (in sec)
    tau_max : float
        max tau value that can be assigned
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
                 tau,
                 tau_max,
                 lam,
                 boundary_conditions,
                 optimizer_cls,
                 optimizer_params=None):
        """Instantiate the ARZ model object.

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
            maximum density term in the model (in veh/m)
        rho_max_max : float
            maximum possible density of the network (in veh/m)
        v_max : float
            initial speed limit of the model. If not actions are provided
            during the simulation procedure, this value is kept constant
            throughout the simulation
        v_max_max : float
            max speed limit that the network can be assigned
        tau : float
            time needed to adjust the velocity of a vehicle from its current
            value to the equilibrium speed (in sec)
        tau_max : float
            max tau value that can be assigned
        lam : float
            exponent of the Green-shield velocity function
        boundary_conditions : str
            conditions at road left and right ends; should either dict or
            string ie. {'constant_both': ((density, speed),(density, speed) )},
            constant value of both ends loop, loop edge values as a ring
            extend_both, extrapolate last value on both ends
        optimizer_cls : type [ mbrl_traffic.utils.optimizers.base.Optimizer ]
            the optimizer class to use when training the model parameters
        optimizer_params : dict
            optimizer-specific parameters
        """
        super(ARZModel, self).__init__(
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
        self.tau = tau
        self.tau_max = tau_max
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
                the model parameters, of the form: [rho_max, v_max, tau, lam]

            Returns
            -------
            float
                the loss associated with the model parameters
            """
            # Store the original model parameters.
            orig_rho_max = deepcopy(self.rho_max)
            orig_v_max = deepcopy(self.v_max)
            orig_tau = deepcopy(self.tau)
            orig_lam = deepcopy(self.lam)

            # Update the model parameter based on the input.
            self.rho_max = x[0]
            self.v_max = x[1]
            self.tau = x[2]
            self.lam = x[3]

            # Sample a batch of state/action/next_state from the replay buffer.
            states, actions, next_states, _, _ = self.replay_buffer.sample()

            # Compute the loss.
            loss = self.compute_loss(states, actions, next_states)

            # Set the model parameters back to their original values.
            self.rho_max = orig_rho_max
            self.v_max = orig_v_max
            self.tau = orig_tau
            self.lam = orig_lam

            return loss

        # Create the optimizer object.
        self.optimizer = optimizer_cls(
            param_low=[0, 0, 0, 0],
            param_high=[rho_max_max, v_max_max, tau_max, LAM_MAX],
            fitness_fn=fitness_fn,
            verbose=verbose,
            **self.optimizer_params
        )

        # critical density defined by the Green-shield Model
        self.rho_crit = self.rho_max / 2

    def initialize(self):
        """See parent class."""
        pass

    def get_next_obs(self, obs, action):
        """See parent class."""
        del action  # actions are not currently used

        # Extract the densities and relative speed.
        rho_t = obs[:int(obs.shape[0]/2)]
        v_t = obs[int(obs.shape[0]/2):]
        q_t = self._relative_flow(rho_t, v_t)

        # Advance the state of the simulation by one step.
        rho_tp1, q_tp1 = self._arz_solve(rho_t, q_t)

        # Compute the speed at the current time step.
        v_tp1 = self.u(rho_tp1, q_tp1)

        # Compute the new observation.
        return np.concatenate((rho_tp1, v_tp1))

    def _relative_flow(self, density, velocity):
        """Calculate actual relative flow from density and velocity.

        Parameters
        ----------
        density : array_like
            density data at every specified point on road
        velocity : array_like
            velocity at every specified point on road

        Returns
        -------
        array_like
            relative flow at every specified point on road
        """
        return (velocity - self._v_eq(density)) * density

    def _v_eq(self, rho):
        """Implement the 'Greenshields model for the equilibrium velocity'.

        Fan, Shimao et al. “Comparative model accuracy of a data-fitted
        generalized Aw-Rascle-Zhang model.” NHM 9 (2014): 239-268.

        Parameters
        ----------
        rho : array_like
            density data at every specified point on road

        Returns
        -------
        array_like
            equilibrium velocity at every specified point on road
        """
        return self.v_max * ((1 - (rho / self.rho_max)) ** self.lam)

    def u(self, rho, q):
        """Calculate actual velocity from density and relative flow.

        Parameters
        ----------
        rho : array_like
            density data at every specified point on road
        q : array_like
            relative flow data at every specified point on road

        Returns
        -------
        array_like
            velocity at every specified point on road
        """
        # avoid division by zero error
        mask = rho == 0
        rho[mask] = 0.0000000001

        return (q / rho) + self._v_eq(rho)

    def _arz_solve(self, rho_t, q_t):
        """Implement Godunov Semi-Implicit scheme for multi-populations.

        Fan, Shimao et al. “Comparative model accuracy of a data-fitted
        generalized Aw-Rascle-Zhang model.” NHM 9 (2014): 239-268.

        Parameters
        ----------
        rho_t : array_like
            density data points on the road length
        q_t : array_ike
           relative flow data points on the road length

        Returns
        -------
        array_like
            next density data points as calculated by the Semi-Implicit Godunov
            scheme
        array_like
            next relative flow data points as calculated by the Semi-Implicit
            Godunov scheme
        """
        # Compute the flux.
        fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half = \
            self._compute_flux(rho_t, q_t)

        # update new points
        new_points = self.arz_update_points(
            fp_higher_half,
            fp_lower_half,
            fy_higher_half,
            fy_lower_half,
            rho_t,
            q_t,
        )

        # Update loop boundary conditions for ring-like experiment.
        if self.boundary_conditions == "loop":
            boundary_left = rho_t[-1], q_t[-1]
            boundary_right = rho_t[-2], q_t[-2]
            rho_tp1 = np.insert(
                np.append(new_points[0], boundary_right[0]),
                0,
                boundary_left[0]
            )
            q_tp1 = np.insert(
                np.append(new_points[1], boundary_right[1]),
                0,
                boundary_left[1]
            )

        # Update boundary conditions by extending/extrapolating boundaries
        # (reduplication).
        elif self.boundary_conditions == "extend_both":
            boundary_left = (new_points[0][0], new_points[1][0])
            boundary_right = (new_points[0][-1], new_points[1][-1])
            rho_tp1 = np.insert(
                np.append(new_points[0], boundary_right[0]),
                0,
                boundary_left[0]
            )
            q_tp1 = np.insert(
                np.append(new_points[1], boundary_right[1]),
                0,
                boundary_left[1]
            )

        # Update boundary conditions by keeping boundaries constant.
        elif isinstance(self.boundary_conditions, dict) and \
                list(self.boundary_conditions.keys())[0] == "constant_both":
            boundary_left = self.boundary_conditions["constant_both"][0]
            boundary_right = self.boundary_conditions["constant_both"][1]
            rho_tp1 = np.insert(
                np.append(new_points[0], boundary_right[0]),
                0,
                boundary_left[0]
            )
            q_tp1 = np.insert(
                np.append(new_points[1], boundary_right[1]),
                0,
                boundary_left[1]
            )

        else:
            raise ValueError("Unknown boundary condition: {}".format(
                self.boundary_conditions))

        return rho_tp1, q_tp1

    def _compute_flux(self, rho_t, q_t):
        """Implement the Flux Supply and Demand Model for flux function.

        'Lebacque, Jean-Patrick & Haj-Salem, Habib & Mammar, Salim. (2005).
        Second order traffic flow modeling: supply-demand analysis of the
        inhomogeneous riemann problem and of boundary conditions'

        Parameters
        ----------
        rho_t : array_like
            density data points on the road length with boundary conditions
        q_t : array_ike
           relative flow data points on the road length with boundary
           conditions

        Returns
        -------
        array_like
            density flux at right boundary of each cell
        array_like
            density flux at left boundary of each cell
        array_like
            relative flow flux at right boundary of each cell
        array_like
            relative flow flux at left boundary of each cell
        """
        rho_crit = self.rho_crit
        v_eq = self._v_eq(rho_t)
        q_max = ((rho_crit * self._v_eq(rho_crit)) + q_t)

        # demand
        d = (rho_t * v_eq + q_t) * (rho_t < rho_crit) \
            + q_max * (rho_t >= rho_crit)

        # supply
        s = (rho_t * v_eq + q_t) * (rho_t > rho_crit) \
            + q_max * (rho_t <= rho_crit)
        s = np.append(s[1:], s[-1])

        # flow flux
        q = np.minimum(d, s)

        # relative flux
        p = q * (q_t / rho_t)

        # fluxes at left cell boundaries
        qm = np.append(q[0], q[:-1])
        pm = np.append(p[0], p[:-1])

        return q, qm, p, pm

    def arz_update_points(self,
                          fp_higher_half,
                          fp_lower_half,
                          fy_higher_half,
                          fy_lower_half,
                          rho_t,
                          q_t):
        """Update our current density and relative flow values.

        Parameters
        ----------
        fp_higher_half : array_like
            density flux at right boundary of each cell
        fp_lower_half : array_like
            density flux at left boundary of each cell
        fy_higher_half : array_like
            relative flow flux at right boundary of each cell
        fy_lower_half : array_like
            relative flow flux at left boundary of each cell
        rho_t : array_like
            current values for density at each point on the road (midpoint of
            cell)
        q_t : array_like
            current values for relative flow at each point on the road
            (midpoint of cell)

        Returns
        -------
        array_like
            next density values at each point on the road
        array_like
            next relative flow values at each point on the road
        """
        # time and cell step
        step = self.dt / self.dx  # where are we referencing this?

        # updating density
        rho_tp1 = rho_t + (step * (fp_lower_half - fp_higher_half))

        # updating relative flow
        # right hand side constant -> we use fsolve to find our roots
        rhs = q_t + (step * (fy_lower_half - fy_higher_half)) \
            + (self.dt / self.tau) * rho_tp1 * self._v_eq(rho_tp1)
        q_tp1 = fsolve(self.myfun, q_t, args=(self.tau, self.v_max,
                                              self.rho_max, rho_tp1, rhs))
        rho_tp1 = rho_tp1[1:-1]
        q_tp1 = q_tp1[1:-1]

        return rho_tp1, q_tp1

    def myfun(self, q_tp1, *args):
        """Help fsolve update our relative flow data.

        Parameters
        ----------
        q_tp1 : array_like
            array whose values to be determined
        args :  v_max : see parent class
                rho_max : see parent class
                cfl : see parent class
                tau : see parent class
                rho_tp1 : see parent class
                rhs : array_like
                    run hand side of function

        Returns
        -------
        array_like or tuple
            functions to be minimized/maximized based on initial values
        """
        tau, v_max, rho_max, rho_tp1, rhs = args

        return q_tp1 + \
            ((self.dt / self.tau) * (rho_tp1 * self.u(rho_tp1, q_tp1)) - rhs)

    def update(self):
        """See parent class."""
        model_params, error = self.optimizer.solve()

        # Update the model parameters in the class.
        self.rho_max = model_params[0]
        self.v_max = model_params[1]
        self.tau_max = model_params[2]
        self.lam = model_params[3]

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
            "tau": self.tau,
            "tau_max": self.tau_max,
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
        self.tau = data["tau"]
        self.tau_max = data["tau_max"]
        self.lam = data["lam"]
        self.boundary_conditions = data["boundary_conditions"]
