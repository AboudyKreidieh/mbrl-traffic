"""Script containing the non-local model object."""
# from mbrl_traffic.models.base import Model
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


class NonLocalModel():
    """Non-local model object

    Keimer, Alexander, Lukas Pflug, and Michele Spinola.
    "Nonlocal scalar conservation laws on bounded domains and applications in traffic flow." 
    SIAM Journal on Mathematical Analysis 50.6 (2018): 6271-6306.
    
    Keimer, Alexander, and Lukas Pflug. "Existence, uniqueness and regularity 
    results on nonlocal balance laws." Journal of Differential Equations 263.7 
    (2017): 4023-4069.

    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 replay_buffer,
                 verbose,
                 dt,
                 rho_max,
                 rho_max_max,
                 v_max,
                 v_max_max,
                 stream_model,
                 lam,
                 l,
                 tfinal,
                 q_init,
                 eta,
                 optimizer_cls,
                 optimizer_params):


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
        l : float
            length of raod
        tfinal : float
            time horizon (in seconds)
        q_init : array_like
            Initial Densities
        eta : float
            looking-ahead parameter
        optimizer_cls : type [ mbrl_traffic.utils.optimizers.base.Optimizer ]
            the optimizer class to use when training the model parameters
        optimizer_params : dict
            optimizer-specific parameters
        """
        super(NonLocalModel, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            replay_buffer=replay_buffer,
            verbose=verbose,
        )

        # discretization points
        Nx = len(q_init)
        self.v_max_new = v_max / l
        self.v_max_max_new = v_max_max / l
        self.x0 = np.linspace(0, 1, Nx)
        self.dx = np.mean(np.gradient(self.x0))
        self.xfine = np.linspace(0, 1, Nx)
        self.ti = []
        self.tfinal = tfinal
        self.dt = dt
        self.q_init = q_init

        # anti derivative of initial density
        self.q_anti = cumtrapz(self.q_init, self.x0, initial=0)
        # finite differences of the anti derivative
        self.q = np.gradient(self.q_anti)

        self.x = self.x0
        self.t = 0
        self.eta = eta
        self.rho_max = rho_max
        self.lam = lam
        self.rho_max_max = rho_max_max

        # Create the optimizer object.
        self.optimizer = optimizer_cls(
            param_low=None,  # FIXME
            param_high=None,  # FIXME
            fitness_fn=None,  # FIXME
            verbose=verbose,
            **optimizer_params
        )

        # critical density defined by the Green-shield Model
        self.rho_crit = self.rho_max / 2

    def initialize(self):
        """See parent class."""
        raise NotImplementedError

    def reset(self):
        """ Reset q and x

            Returns
            -------
            array_like
                the initial observation of the space
            """
        self.x = self.x0
        # finite differences of the anti derivative
        self.q = np.gradient(self.q_anti)

        return self.q, self.x

    def get_next_obs(self, action=None):
        """ See parent class"""

        densities = self.compute_nonlocal()
        velocities = self.vel(densities, self.rho_max, self.v_max_new, self.lam)

        return densities, velocities

    def update(self):
        """See parent class."""
        raise NotImplementedError

    def compute_loss(self, states, actions, next_states):
        """See parent class."""
        # Compute the predicted next states.
        # expected_next_states = self.get_next_obs(states, actions)

        # Compute the loss.
        return None  # FIXME

    def get_td_map(self):
        """See parent class."""
        raise NotImplementedError

    def save(self, save_path):
        """See parent class."""
        raise NotImplementedError

    def load(self, load_path):
        """See parent class."""
        raise NotImplementedError

    def compute_nonlocal(self):

        """Compute next density values

       Returns
        -------
        array_like
              next density values
        """
        while self.x[-2] >= 1:
            self.x = np.delete(self.x, -1)
            self.q = np.delete(self.q, -1)

        while self.x[0] >= self.dx:
            self.x = np.append(self.x[0] - self.dx, self.x)
            left_point = self.q[-1] / np.diff(self.x[-2:])
            self.q = np.append(left_point * np.diff(self.x[0:2]), self.q)

        w = self.integrate_nonlocal_term(self.q, self.x)
        self.x = self.x + self.dt * self.vel(w, self.rho_max, self.v_max_new, self.lam)

        # update boundary conditions here
        while min(np.gradient(self.x)) < 1e-6:
            ind = np.argmin(np.gradient(self.x))  # find index
            if ind > 1:
                mq = self.q[ind - 1] + self.q[ind]
                self.x = np.delete(self.x, ind)
                self.q[ind - 1] = mq
                self.q = np.delete(self.q, ind)
            else:
                mq = self.q[ind + 1] + self.q[ind]
                self.x = np.delete(self.x, ind + 1)
                self.q[ind] = mq
                self.q = np.delete(self.q, ind + 1)

        old_x = self.x + np.diff(self.x[0:2]) / 2
        current_density = self.q / np.gradient(self.x)

        while old_x[-2] > 1:
            old_x = np.delete(old_x, -1)
            current_density = np.delete(current_density, -1)

        new_x = self.xfine
        set_interp = interp1d(np.append(0, old_x), np.append(current_density[-1], current_density), kind='nearest')
        new_density = set_interp(new_x)

        return new_density

    def integrate_nonlocal_term(self, q, x):
        """Numerical Integration of Non-local Term

        Parameters
        ----------
        See Parent class

        Returns
        -------
        array_like
            Non-local terms at every point in the road
        """

        x_size = len(x)
        q = np.append(q, q[1:])

        n = 0
        while len(x) < (2 * x_size) - 1:
            x= np.append(x, x[-1] + np.diff(x[n:n + 2]))
            n += 1
        dens = q / np.gradient(x)

        cc = dens[:x_size - 2]
        cc2 = dens[x_size:]
        d1 = np.append(cc, cc2)
        dens = np.append(d1, cc[0:2])

        a_t = self.a(x).reshape(len(x), 1)
        b_t = self.b(x, self.eta).reshape(len(x), 1)
        upbnd = np.maximum(np.minimum(x[1:], b_t), a_t)
        lobnd = np.minimum(np.maximum(x[:-1], a_t), b_t)

        part_a = self.gamma_y(np.matlib.repmat(x.reshape(len(x), 1), 1, len(x) - 1), upbnd, self.eta)
        part_b = self.gamma_y(np.matlib.repmat(x.reshape(len(x), 1), 1, len(x) - 1), lobnd, self.eta)
        w_1 = np.sum(np.multiply(dens[:-1], part_a - part_b), 1)

        return w_1[0:x_size]

    def a(self,x):
        """Finding the lower bound of the area of of integration

        Parameters
        ----------
        x : See parent class

       Returns
        -------
        array_like
            lower bound of the area of of integration"""

        return np.minimum(x, 1)

    def b(self, x, eta):
        """Finding the lower bound of the area of of integration.\
            We include eta for the look ahead parameter

        Parameters
        ----------
        x : See parent class
        eta : See parent class

       Returns
        -------
        array_like
            lower bound of the area of of integration """

        return x + eta

    def gamma_y(self, x, y, eta):
        """ Weight of the non local term

        Parameters
        ----------
        y : bound (either upper or lower)
        x : See parent class
        eta: See parent class

       Returns
        -------
        array_like
            Weight of the non local term"""

        return (2 * (y - x) - (y - x) ** 2 / eta) / eta

    def vel(self, w, rho_max, v_max_new, lam):
        """Implement the Greenshields model for the equilibrium velocity.

               Greenshields, B. D., Ws Channing, and Hh Miller. "A study of traffic
               capacity." Highway research board proceedings. Vol. 1935. National
               Research Council (USA), Highway Research Board, 1935.

        Parameters
        ----------
        w: See parent class
        rho_max: See parent class
        v_max_new: See parent class
        lam: See parent class

        Returns
        -------
        array_like
            velocity at every specified point on road
        """

        return v_max_new * (1 - w / rho_max) ** lam


if __name__ == "__main__":
    tn = 1001
    tfinal = 20
    # dt = tfinal / tn
    dt = 0.1
    eta = 0.01
    rho_max = 1
    v_max = 11
    l = 260
    q_init = np.array([0.536000000000000,
                  0.541000000000000, 0.546000000000000, 0.551000000000000, 0.555000000000000, 0.555000000000000,
                  0.555000000000000, 0.555000000000001, 0.554999999999999, 0.555000000000000, 0.555000000000000,
                  0.555000000000000, 0.555000000000001, 0.554999999999999, 0.555000000000000, 0.555000000000000,
                  0.555000000000000, 0.555000000000001, 0.554999999999999, 0.555000000000000, 0.555000000000000,
                  0.555000000000000, 0.555000000000001, 0.554999999999999, 0.555000000000000, 0.555000000000000,
                  0.555000000000000, 0.555000000000001, 0.553999999999999, 0.549000000000000, 0.544000000000000,
                  0.539000000000000, 0.534000000000000, 0.528999999999999, 0.524000000000000, 0.519000000000000,
                  0.514000000000000, 0.509000000000000, 0.503999999999999, 0.499000000000000, 0.494000000000000,
                  0.489000000000000, 0.484000000000000, 0.478999999999999, 0.474000000000000, 0.469000000000000,
                  0.464000000000000, 0.459000000000000, 0.453999999999999, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.450000000000000, 0.450000000000000, 0.454500000000000, 0.459500000000000,
                  0.464500000000000, 0.469500000000000, 0.474500000000000, 0.479500000000000, 0.484500000000000,
                  0.489500000000000, 0.494500000000000, 0.499500000000000, 0.504500000000000, 0.509500000000000,
                  0.514500000000000, 0.519500000000000, 0.524500000000000, 0.529500000000000, 0.534500000000000,
                  0.539500000000000, 0.544500000000000, 0.549500000000000, 0.554500000000000, 0.559500000000000,
                  0.564500000000000, 0.569500000000000, 0.574500000000000, 0.579500000000000, 0.584500000000000,
                  0.589500000000000, 0.594500000000000, 0.599500000000000, 0.604500000000000, 0.609500000000000,
                  0.614500000000000, 0.618000000000000, 0.618000000000000, 0.618000000000000, 0.618000000000000,
                  0.618000000000000, 0.618000000000000, 0.618000000000000, 0.618000000000000, 0.618000000000000,
                  0.618000000000000, 0.618000000000000, 0.618000000000000, 0.613500000000000, 0.608500000000000,
                  0.603500000000000, 0.598500000000000, 0.593500000000000, 0.588500000000000, 0.583500000000000,
                  0.578500000000000, 0.573500000000000, 0.568500000000000, 0.563500000000000, 0.558500000000000,
                  0.553500000000000, 0.548500000000000, 0.543500000000000, 0.538500000000000, 0.533500000000000,
                  0.528500000000000, 0.523500000000000, 0.518500000000000, 0.513500000000000, 0.508500000000000,
                  0.503500000000000, 0.498500000000000, 0.493500000000000, 0.488500000000000, 0.483500000000000,
                  0.478500000000000, 0.473500000000000, 0.468500000000000, 0.463500000000000, 0.458500000000000,
                  0.453500000000000, 0.450000000000000, 0.454500000000000, 0.459500000000000, 0.464500000000000,
                  0.469500000000000, 0.474500000000000, 0.479500000000000, 0.484500000000000, 0.489500000000000,
                  0.494500000000000, 0.499500000000000, 0.504500000000000, 0.509500000000000, 0.514500000000000,
                  0.519500000000000, 0.524500000000000, 0.529500000000000, 0.534500000000000, 0.539500000000000,
                  0.544500000000000, 0.549500000000000, 0.554500000000000, 0.559500000000000, 0.564500000000000,
                  0.569500000000000, 0.574500000000000, 0.579500000000000, 0.584500000000000, 0.589500000000000,
                  0.594500000000000, 0.599500000000000, 0.604500000000000, 0.609500000000000, 0.614500000000000,
                  0.619500000000000, 0.624500000000000, 0.629500000000000, 0.634500000000000, 0.639500000000000,
                  0.644500000000000, 0.649500000000000, 0.653500000000000, 0.653500000000000, 0.653500000000000,
                  0.653500000000000, 0.653500000000000, 0.649000000000000, 0.644000000000000, 0.639000000000000,
                  0.634000000000000, 0.629000000000000, 0.624000000000000, 0.619000000000000, 0.614000000000000,
                  0.609000000000000, 0.604000000000000, 0.599000000000000, 0.594000000000000, 0.589000000000000,
                  0.584000000000000, 0.579000000000000, 0.574000000000000, 0.569000000000000, 0.564000000000000,
                  0.559000000000000, 0.554000000000000, 0.549000000000000, 0.544000000000000, 0.539000000000000,
                  0.534000000000000, 0.529000000000000, 0.524000000000000, 0.519000000000000, 0.514000000000000,
                  0.509000000000000, 0.504000000000000, 0.499000000000000, 0.494000000000000, 0.489000000000000,
                  0.484000000000000, 0.479000000000000, 0.476000000000000, 0.476000000000000, 0.476000000000000,
                  0.476000000000000, 0.476000000000000, 0.477000000000000, 0.482000000000000, 0.487000000000000,
                  0.492000000000000, 0.497000000000000, 0.502000000000000, 0.507000000000000, 0.512000000000000,
                  0.517000000000000, 0.522000000000000, 0.527000000000000, 0.532000000000000, 0.537000000000000,
                  0.542000000000000, 0.547000000000000, 0.552000000000000, 0.557000000000000, 0.562000000000000,
                  0.567000000000000, 0.572000000000000, 0.577000000000000, 0.582000000000000, 0.587000000000000,
                  0.592000000000000, 0.597000000000000, 0.602000000000000, 0.607000000000000, 0.612000000000000,
                  0.617000000000000, 0.622000000000000, 0.627000000000000, 0.632000000000000, 0.637000000000000,
                  0.642000000000000, 0.647000000000000, 0.652000000000000, 0.652500000000000, 0.652500000000000,
                  0.652500000000000, 0.652500000000000, 0.650500000000000, 0.645500000000000, 0.640500000000000,
                  0.635500000000000, 0.630500000000000, 0.625500000000000, 0.620500000000000, 0.615500000000000,
                  0.610500000000000, 0.605500000000000, 0.600500000000000, 0.595500000000000, 0.590500000000000,
                  0.585500000000000, 0.580500000000000, 0.575500000000000, 0.570500000000000, 0.565500000000000,
                  0.560500000000000, 0.555500000000000, 0.550500000000000, 0.545500000000000, 0.540500000000000,
                  0.535500000000000, 0.530500000000000, 0.525500000000000, 0.520500000000000, 0.515500000000000,
                  0.510500000000000, 0.505500000000000, 0.500500000000000, 0.495500000000000, 0.490500000000000,
                  0.485500000000000, 0.480500000000000, 0.475500000000000, 0.470500000000000, 0.465500000000000,
                  0.460500000000000, 0.455500000000000, 0.454500000000000, 0.459000000000000, 0.464000000000000,
                  0.469000000000000, 0.474000000000000, 0.479000000000000, 0.484000000000000, 0.489000000000000,
                  0.494000000000000, 0.499000000000000, 0.504000000000000, 0.509000000000000, 0.514000000000000,
                  0.519000000000000, 0.524000000000000, 0.529000000000000, 0.534000000000000, 0.539000000000000,
                  0.544000000000000, 0.549000000000000, 0.554000000000000, 0.559000000000000, 0.564000000000000,
                  0.569000000000000, 0.574000000000000, 0.579000000000000, 0.584000000000000, 0.589000000000000,
                  0.594000000000000, 0.599000000000000, 0.604000000000000, 0.609000000000000, 0.614000000000000,
                  0.619000000000000, 0.624000000000000, 0.624500000000000, 0.624500000000000, 0.624500000000000,
                  0.624500000000000, 0.624500000000000, 0.624500000000000, 0.624500000000000, 0.624500000000000,
                  0.624500000000000, 0.624500000000000, 0.620500000000000, 0.615500000000000, 0.610500000000000,
                  0.605500000000000, 0.600500000000000, 0.595500000000000, 0.590500000000000, 0.585500000000000,
                  0.580500000000000, 0.575500000000000, 0.570500000000000, 0.565500000000000, 0.560500000000000,
                  0.555500000000000, 0.550500000000000, 0.545500000000000, 0.540500000000000, 0.535500000000000,
                  0.530500000000000, 0.525500000000000, 0.520500000000000, 0.515500000000000, 0.510500000000000,
                  0.505500000000000, 0.500500000000000, 0.495500000000000, 0.490500000000000, 0.485500000000000,
                  0.480500000000000, 0.475500000000000, 0.470500000000000, 0.465500000000000, 0.460500000000000,
                  0.455500000000000, 0.450500000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.454000000000000, 0.459000000000000, 0.464000000000000, 0.469000000000000,
                  0.474000000000000, 0.479000000000000, 0.484000000000000, 0.489000000000000, 0.494000000000000,
                  0.499000000000000, 0.504000000000000, 0.509000000000000, 0.514000000000000, 0.519000000000000,
                  0.524000000000000, 0.529000000000000, 0.534000000000000, 0.539000000000000, 0.544000000000000,
                  0.549000000000000, 0.554000000000000, 0.559000000000000, 0.564000000000001, 0.569000000000000,
                  0.574000000000000, 0.579000000000000, 0.582000000000000, 0.582000000000000, 0.582000000000000,
                  0.582000000000000, 0.582000000000000, 0.582000000000000, 0.582000000000000, 0.582000000000000,
                  0.582000000000000, 0.582000000000000, 0.582000000000000, 0.582000000000000, 0.582000000000000,
                  0.582000000000000, 0.582000000000000, 0.582000000000000, 0.582000000000000, 0.582000000000000,
                  0.582000000000000, 0.578000000000000, 0.573000000000000, 0.568000000000000, 0.563000000000000,
                  0.558000000000000, 0.553000000000000, 0.548000000000000, 0.543000000000000, 0.538000000000000,
                  0.533000000000000, 0.528000000000000, 0.523000000000000, 0.518000000000000, 0.513000000000000,
                  0.508000000000000, 0.503000000000000, 0.498000000000000, 0.493000000000000, 0.488000000000000,
                  0.483000000000000, 0.478000000000000, 0.473000000000000, 0.468000000000000, 0.463000000000000,
                  0.458000000000000, 0.453000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000,
                  0.450000000000000, 0.452500000000000, 0.457500000000000, 0.462500000000000, 0.467500000000000,
                  0.472500000000000, 0.477500000000000, 0.482500000000000, 0.487500000000000])
    lam = 1
    qfine = []

    sess = None
    ob_space = None
    ac_space = None
    replay_buffer = None
    verbose = None
    rho_max_max = 1
    v_max_max = 100
    stream_model = None

    mode = NonLocalModel(sess,
                 ob_space,
                 ac_space,
                 replay_buffer,
                 verbose,
                 dt,
                 rho_max,
                 rho_max_max,
                 v_max,
                 v_max_max,
                 stream_model,
                 lam,
                 l,
                 tfinal,
                 q_init,
                 eta)
    ti = []
    t = 0
    for i in np.arange(0, (tfinal / dt)):

        densities, _ = mode.get_next_obs()

        # reset initial conditions
        mode.reset()
        # update eta
        mode.eta = mode.eta - (mode.eta/2)

        t += dt
        if len(qfine) == 0:
            qfine = densities
            ti = np.append(ti, t)
        else:
            qfine = np.vstack((qfine, densities))
            ti = np.append(ti, t)

        if (i > 1) and (np.mod(int(i), 10)) == 0:
            # pass
            # 2d density Plot
            # plt.figure(1)
            # plt.clf()
            # plt.plot(mode.xfine, densities)
            # plt.xlim((0, 1))
            # plt.draw()
            # plt.pause(0.1)

            # Plot densities surface plot
            plt.figure(1)
            plt.clf()
            all_densities = qfine
            plt.contourf(mode.xfine, ti, all_densities, levels=900, cmap='jet')
            plt.colorbar(shrink=0.8)
            plt.ylim((0, tfinal))
            plt.xlim((0, 1))
            plt.draw()
            plt.pause(0.1)
