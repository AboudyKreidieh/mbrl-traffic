import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def non_local():

    # discretization
    Nx = 1001
    Tn = 1001
    x0 = np.linspace(0, 1, Nx + 1)
    dx = np.mean(np.diff(x0))
    Tfinal = 10
    dt = Tfinal / Tn
    Xfine = np.linspace(0, 1, 1001)
    Ti = []
    Qfine = []
    # nonlocal impact
    eta = 0.1
    a = lambda x: np.minimum(x, 1)
    b = lambda x: np.minimum(x + eta, 1)
    GAMMA_Y = lambda t, x, y: (2 * (y - x) - (y - x) ** 2 / eta) / eta

    # initialdatum and boundary conditions
    Q0 = lambda x: 0 * x
    u = lambda t: (t >= 0) * (t < 1) + (t >= 2) * (t < 3) + (t >= 4) * (t < 5) + (t >= 6) * (t < 7) + (t >= 8) * (t < 9)
    v = lambda t: (t >= 1) * (t < 2) + (t >= 3) * (t < 4) + (t >= 5) * (t < 6) + (t >= 7) * (t < 8) + (t >= 9) * (t < 10)

    # velocity function
    vel = lambda w, t, x: 1 - w

    Q = Q0(x0[1:]) - Q0(x0[0:- 1]) # FIX ME
    X = x0
    T = 0

    Nt = Tfinal / dt
    for i in np.arange(1, (Tfinal / dt)):
        # print(i)
        # if Nt > 200:
        #     if (np.mod(i, 1000) == 0):
        #         # print(i / Nt * 100)

        w = integrate_nonlocal_term(GAMMA_Y, a, b, Q, X, T, v, eta)
        X = X + dt * vel(w, T, X)

        while min(np.diff(X)) < 1e-6:
            ind = np.argmin(np.diff(X))
            # print(ind)
            if ind > 1:
                mq = Q[ind - 1] + Q[ind]
                X = np.delete(X, ind)
                Q[ind - 1] = mq
                Q = np.delete(Q, ind)
            else:
                mq = Q[ind + 1] + Q[ind]
                X = np.delete(X, ind+1)
                Q[ind] = mq
                Q = np.delete(Q, ind+1)

        old_x = X[1:] + np.diff(X[0:2]) / 2
        old_y = Q / np.diff(X)
        Xfine[Xfine < min(old_x)] = np.nan
        Xfine[Xfine > max(old_x)] = np.nan
        new_x = Xfine
        set_interp = interp1d(old_x, old_y, kind='nearest')

        new_y = set_interp(new_x)
        # new_y[new_y == np.nan] = 0
        if Qfine == []:
            Qfine = new_y
        else:
            Qfine = np.vstack((Qfine,new_y))

        Ti = Ti + [T]
        # print(i > 1 & np.mod(int(i), 100) == 0)
        if (i > 1) and (np.mod(int(i), 100)) == 0:
            # plt.clf()
            all_densities = Qfine
            x_vector, y_vector = np.meshgrid(Xfine, np.array(Ti))
            plt.contourf(x_vector, y_vector, all_densities, levels=900, cmap='jet')
            plt.colorbar(shrink=0.8)
            # plt.clim(-5, 5)
            plt.ylim((0, 10))
            plt.xlim((0, 1))
            plt.draw()
            plt.pause(0.1)

            # figure(11)
            # plt.plot(X[1:] + np.diff(X[:2]) / 2, Q / np.diff(X))
            # axis([-0.1, 1.1, 0, 1])

            #
            # figure(16)
            # h = surf(Xfine, Ti, vel(Qfine,0, 0))
            # set(h, 'LineStyle', 'none');
            # view(2)
            # caxis([0, 1])
            # drawnow()
            #
            # figure(17)
            # h = surf(Xfine, Ti, Qfine. * vel(Qfine,0, 0))
            # set(h, 'LineStyle', 'none')
            # caxis([0, 1])
            # view(2)
            # drawnow()

        while X[-1] >= 1:
            X = np.delete(X, -1)
            Q = np.delete(Q, -1)

        while X[0] >= dx:
            X = np.append(X[0] - dx, X)
            Q = np.append(u(T) * dx, Q)

        T = T + dt

    Z = X[1:] / 2 + X[0:-1] / 2
    D = X[1:] - X[0:-1]
    XE = X
    QD = Q / D
    QDE = QD
    plt.show()

    return X, Q, Z, D


def integrate_nonlocal_term(GAMMA_Y, a, b, q, x, t, v, eta):

    a_T = a(x).reshape(len(x), 1)
    b_T = b(x).reshape(len(x), 1)
    UpBnd = np.maximum(np.minimum(x[1:], b_T), a_T)
    LoBnd = np.minimum(np.maximum(x[:-1], a_T), b_T)

    part_a = GAMMA_Y(t, np.matlib.repmat(x.reshape(len(x), 1),1,len(x)-1), UpBnd)
    part_b = GAMMA_Y(t, np.matlib.repmat(x.reshape(len(x), 1), 1,len(x)-1), LoBnd)
    part_c = v(t)*(GAMMA_Y(t,x,a(x)+eta) - GAMMA_Y(t,x,b(x)))
    w_1 = np.sum(np.multiply(q / (x[1:] - x[:-1]), part_a - part_b), 1)
    w_2 = w_1 + part_c
    w_2[x >1] = v(t)
    return w_2


def visualize_plots(x, all_densities, all_speeds, time_steps):
    """Create surface plot for density and velocity evolution of simulation.

    Parameters
    ----------
    x : array-like or list
        points of the road length to plot against
    all_densities: N x M array-like matrix
        density values on the road length M at every time step N.
    all_speeds: N x M array-like matrix
        velocity values on the road length M at every time step N.
    time_steps: list
        discrete time steps that the simulation has run for
    """
    # density plot
    fig, plots = plt.subplots(2, figsize=(10, 10))
    fig.subplots_adjust(hspace=.5)
    y_vector, x_vector = np.meshgrid(x, time_steps)
    first_plot = plots[0].contourf(x_vector, y_vector, all_densities, levels=900, cmap='jet')
    plots[0].set(ylabel='Length (Position on Street in meters)', xlabel='Time (seconds)')
    plots[0].set_title('Density Evolution')
    color_bar = fig.colorbar(first_plot, ax=plots[0], shrink=0.8)
    color_bar.ax.set_title('Density\nLevels', fontsize=8)

    # velocity plot
    second_plot = plots[1].contourf(x_vector, y_vector, all_speeds, levels=900, cmap='jet')
    plots[1].set(ylabel='Length (Position on Street in meters)', xlabel='Time (seconds)')
    plots[1].set_title('Velocity Evolution (m/s)')
    color_bar1 = fig.colorbar(second_plot, ax=plots[1], shrink=0.8)
    color_bar1.ax.set_title('Velocity\nLevels (m/s)', fontsize=8)
    plt.show()

non_local()