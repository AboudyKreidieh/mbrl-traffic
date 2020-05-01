"""Script containing the evaluator for pre-trained models."""
import sys
import argparse
import numpy as np
import pandas as pd
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation

from mbrl_traffic.models import LWRModel
from mbrl_traffic.models import ARZModel
# from mbrl_traffic.models import NonLocalModel
from mbrl_traffic.models import NoOpModel
from mbrl_traffic.models import FeedForwardModel
from mbrl_traffic.utils.tf_util import make_session


def parse_args(args):
    """Parse evaluation options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Simulate and evaluate a macroscopic dynamics model.",
        epilog="python evaluate_model.py <results_dir>")

    # required input parameters
    parser.add_argument(
        'results_dir', type=str,
        help='The location of the logged data during the training procedure. '
             'This will contain the model parameters under the checkpoints '
             'folder.')

    # optional input parameters
    parser.add_argument(
        '--checkpoint_num', type=int, default=None,
        help='the checkpoint number. If set to None, the last checkpoint is '
             'used.')
    parser.add_argument(
        '--initial_conditions', type=str, default=None,
        help='The path to a file containing a list of initial conditions to '
             'evaluate from. Each row in the file should constitute an '
             'initial condition, and the rows should be the speed or density '
             'of a specific edge.')
    parser.add_argument(
        '--save_path', type=str, default=None,
        help='The path were the results should be logged to. If this term is '
             'not specified, the plots, videos, and data are stored in the '
             'same directory as "results_dir"')
    parser.add_argument(
        '--steps', type=int, default=1000,
        help='the time horizon of a simulation')
    parser.add_argument(
        '--runs', type=int, default=1,
        help='the number of simulations to perform per initial condition.')
    parser.add_argument(
        '--plot_only', action='store_true',
        help='whether to only perform plotting operations. If set to True, '
             'no simulations are run and the csv files in "results_dir" '
             'ending with with ".csv" are plotted and stored in "save_path".')
    parser.add_argument(
        '--svg', action='store_true',
        help='whether to generate the heatmaps as svg files. If set to False, '
             'they are generated as png files.')

    return parser.parse_known_args(args)[0]


def import_data(fp):
    """Import the speed and density data from a specific file.

    Parameters
    ----------
    fp : str
        the file path to the macroscopic data file

    Returns
    -------
    array_like
        an array of the time associated with each observation
    array_like
        a matrix of speeds and observations for every time step
    """
    df = pd.read_csv(fp)

    # Extract the times.
    times = df["time"]

    # Extract only the speeds and densities (keep out the time).
    num_sections = sum(v.startswith("speed") for v in list(df.columns.values))
    df = df[["speed_{}".format(i) for i in range(num_sections)] +
            ["density_{}".format(i) for i in range(num_sections)]]

    # Return a numpy array version of the above.
    return times.values, df.values


def plot_heatmap(times, obses, save_path, postfix="", svg=False):
    """Generate heatmaps of the speeds, densities, and flows.

    Parameters
    ----------
    times : array_like
        an array of the time associated with each observation
    obses : array_like
        a matrix of speeds and observations for every time step
    save_path : str
        the path were the results should be logged to
    postfix : str
        a postfix to the name of the file to be saved
    svg : bool
        whether to generate the heatmaps as svg files. If set to False, they
        are generated as png files.
    """
    # FIXME: add parameters
    min_speed = 0
    max_speed = 30
    length = 1500
    total_time = times[-1]
    min_density = 0
    max_density = 0.2

    # some plotting parameters
    cdict = {
        'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
        'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
        'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
    }
    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    a = obses[:, :round(obses.shape[1] / 2)].T
    b = obses[:, round(obses.shape[1] / 2):].T
    c = a * b * 3600

    # Plot the average speed plots.
    plt.figure(figsize=(16, 9))
    norm = plt.Normalize(min_speed, max_speed)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Position (m)", fontsize=20)
    plt.imshow(a, extent=(0, total_time, 0, length), origin='lower',
               aspect='auto', cmap=my_cmap, norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('Velocity (m/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    speed_save_path = os.path.join(save_path, "speed_{}.{}".format(
        postfix, "svg" if svg else "png"))
    plt.savefig(speed_save_path, bbox_inches='tight')
    plt.clf()
    plt.close()

    # Plot the density plots.
    plt.figure(figsize=(16, 9))
    norm = plt.Normalize(min_density, max_density)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Position (m)", fontsize=20)
    plt.imshow(b, extent=(0, total_time, 0, length), origin='lower',
               aspect='auto', cmap=my_cmap, norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('Density (veh/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    density_save_path = os.path.join(save_path, "density_{}.{}".format(
        postfix, "svg" if svg else "png"))
    plt.savefig(density_save_path, bbox_inches='tight')
    plt.clf()
    plt.close()

    # Plot the flow plots.
    plt.figure(figsize=(16, 9))
    norm = plt.Normalize(min_speed * min_density * 3600,
                         0.2 * max_speed * max_density * 3600)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Position (m)", fontsize=20)
    plt.imshow(c, extent=(0, total_time, 0, length), origin='lower',
               aspect='auto', cmap=my_cmap, norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('flow (veh/hr)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    flow_save_path = os.path.join(save_path, "flow_{}.{}".format(
        postfix, "svg" if svg else "png"))
    plt.savefig(flow_save_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def generate_videos(times, obses, save_path, postfix=""):
    """Generate videos of the evolution of average speeds and densities.

    Parameters
    ----------
    times : array_like
        an array of the time associated with each observation
    obses : array_like
        a matrix of speeds and observations for every time step
    save_path : str
        the path were the results should be logged to
    postfix : str
        a postfix to the name of the file to be saved
    """
    def update_line(num, data, line):
        x = np.arange(data.shape[1]) / data.shape[1]
        # line.set_data(np.concatenate(([x], [data[num, :]]), axis=1))
        line.set_xdata(x)
        line.set_ydata(data[num, :])
        return line,

    # FIXME: add parameters
    max_speed = 30
    max_density = 0.2
    max_steps = 600
    speeds = obses[:, :round(obses.shape[1] / 2)]
    densities = obses[:, round(obses.shape[1] / 2):]

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

    # Video for the average speeds.
    fig1 = plt.figure(figsize=(16, 9))
    l, = plt.plot([], [], 'b-', lw=2)
    plt.xlim(0, 1)
    plt.ylim(0, max_speed)
    plt.xlabel('Section', fontsize=20)
    plt.ylabel('Average Speed (m/s)', fontsize=20)
    line_ani = animation.FuncAnimation(
        fig1, update_line, min(max_steps, times.shape[0]),
        fargs=(speeds, l,), interval=times.shape[0], blit=True)
    line_ani.save(
        os.path.join(save_path, 'speed_{}.mp4'.format(postfix)),
        writer=writer)
    plt.close()

    # Video for the densities.
    fig2 = plt.figure()
    l, = plt.plot([], [], 'b-', lw=2)
    plt.xlim(0, 1)
    plt.ylim(0, max_density)
    plt.xlabel('Section', fontsize=20)
    plt.title('Density (veh/m)', fontsize=20)
    line_ani = animation.FuncAnimation(
        fig2, update_line, min(max_steps, times.shape[0]),
        fargs=(densities, l,), interval=times.shape[0], blit=True)
    line_ani.save(
        os.path.join(save_path, 'density_{}.mp4'.format(postfix)),
        writer=writer)
    plt.close()


def get_model_cls(model):
    """Get the model class from the name of the model.

    Parameters
    ----------
    model : str
        the name of the model

    Returns
    -------
    type [ mbrl_traffic.models.base.Model ]
        the model class
    """
    if model == "LWRModel":
        return LWRModel
    elif model == "ARZModel":
        return ARZModel
    # elif model == "NonLocalModel":
    #     return NonLocalModel
    elif model == "NoOpModel":
        return NoOpModel
    elif model == "FeedForwardModel":
        return FeedForwardModel
    else:
        raise ValueError("Unknown model: {}".format(model))


def get_model_ckpt(base_dir, ckpt_num, model):
    """Get the path to the model checkpoint.

    Parameters
    ----------
    base_dir : str
        the location of the logged data during the training procedure.
    ckpt_num : int
        the checkpoint number. If set to None, the last checkpoint is used.
    model : str
        the name of the model

    Returns
    -------
    str
        the path to the model checkpoint
    """
    model_ckpt = os.path.join(base_dir, "checkpoints")

    if ckpt_num is None:
        # Get the last checkpoint number.
        filenames = os.listdir(model_ckpt)
        metanum = [int(f.split("-")[-1]) for f in filenames]
        ckpt_num = max(metanum)

    # Add the iteration number.
    model_ckpt = os.path.join(model_ckpt, "itr-{}".format(ckpt_num))

    # Add the file path of the checkpoint.
    if model == "FeedForwardModel":
        model_ckpt = os.path.join(model_ckpt, "model")
    elif model == "NoOpModel":
        model_ckpt = None  # no checkpoint needed
    else:
        model_ckpt = os.path.join(model_ckpt, "model.json")

    return model_ckpt


def main(args):
    """Run the evaluation operation."""
    flags = parse_args(args)
    save_path = flags.save_path or flags.results_dir

    if flags.plot_only:
        # files ending with .csv
        macro_fp = os.listdir(flags.results_dir)
        macro_fp = [os.path.join(flags.results_dir, fp)
                    for fp in sorted(macro_fp) if fp.endswith(".csv")]

        for i, fp in enumerate(macro_fp):
            # Import the data from the log path.
            times, obses = import_data(fp)

            # Generate the heatmaps of the speeds, flows, and densities.
            plot_heatmap(
                times, obses, save_path, postfix="{}".format(i), svg=flags.svg)

            # Generate the videos of the speeds and densities.
            generate_videos(
                times, obses, save_path, postfix="{}".format(i))

    else:
        # Get the model type and its parameters from the hyperparameters.json
        # file.
        with open(os.path.join(args.results_dir, 'hyperparameters.json')) as f:
            params = json.load(f)
            model_cls = get_model_cls(params["model"])
            model_params = params["model_params"]
            model_ckpt = get_model_ckpt(
                flags.results_dir, flags.checkpoint_num, params["model"])

        graph = tf.Graph()
        with graph.as_default():
            # Create a tensorflow session.
            sess = make_session(num_cpu=3, graph=graph)

            # Recreate the model.
            model = model_cls(
                sess=sess,
                ob_space=None,
                ac_space=None,
                replay_buffer=None,
                verbose=2,
                **model_params
            )

            # Initialize everything.
            sess.run(tf.compat.v1.global_variables_initializer())
            model.load(model_ckpt)

        # Import the initial conditions.
        initial_conditions = np.genfromtxt(
            flags.initial_conditions,
            dtype=np.float, delimiter=',', skip_header=1)

        # Loop through the initial conditions.
        for i in range(initial_conditions.shape[0]):
            init = initial_conditions[i, :]
            sections = round(len(init) / 2)

            # Loop through number of runs.
            for j in range(flags.runs):
                # Import initial conditions.
                times = [0]
                obses = [init]

                # Run the simulation for the given horizon.
                with sess.as_default(), graph.as_default():
                    for t in range(flags.steps):
                        obs_tp1 = model.get_next_obs(obses[-1], action=None)
                        obses.append(obs_tp1)
                        times.append(t + 1)  # FIXME: use dt

                # Save simulation data.
                header = ["time"] + \
                    ["speed_{}".format(k) for k in range(sections)] + \
                    ["density_{}".format(k) for k in range(sections)]

                np.savetxt(
                    os.path.join(save_path, "macro_{}_{}.csv".format(i, j)),
                    np.concatenate(([times], obses), axis=1),
                    delimiter=",",
                    header=",".join(header))

                # Generate the heatmaps of the speeds, flows, and densities.
                plot_heatmap(times, obses, save_path,
                             postfix="{}_{}".format(i, j), svg=flags.svg)

                # Generate the videos of the speeds and densities.
                generate_videos(times, obses, save_path,
                                postfix="{}_{}".format(i, j))


if __name__ == "__main__":
    main(sys.argv[1:])
