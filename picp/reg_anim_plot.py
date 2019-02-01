from itertools import cycle
from math import sqrt, acos, sin, cos

import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Arrow
import matplotlib.pyplot as plt

from picp.util.geometry import apply_tf_on_point_cloud
from python.pypm.icp import Penalty

SCANS_ID_TO_COLOR = ["blue", "red", "green", "orange", "magenta"]
# SCANS_ID_TO_COLOR = cm.get_cmap('tab20b').colors
# SCANS_ID_TO_COLOR = cm.get_cmap('Dark2').colors


def draw_elipse(ax, avg, cov, color, label=None):
    λ, v = np.linalg.eig(cov)
    sort_indices = np.argsort(λ)[::-1]
    λ = λ[sort_indices]
    v = v[sort_indices]
    # The angle is the angle of the eigen vector with the largest eigen value
    angle = np.rad2deg(np.arctan2(v[0, 1].real, v[0, 0].real))
    # angle = np.rad2deg(np.arccos(v[0, 0]))
    ell =  Ellipse(xy=avg,
                   width=2*sqrt(λ[0]), # λ[0] == σ²
                   height=2*sqrt(λ[1]),
                   angle=-angle,
                   edgecolor=color,
                   label=label)
    ell.set_facecolor('none')
    if label is not None:
        ax.add_patch(ell)
    else:
        ax.add_artist(ell)
    return [ell]

class ExperimentationPlot:

    def __init__(self, label, ax, trajectories, penalties):
        self.ax = ax
        self.label = label
        self.trajectories = trajectories
        self.penalties = penalties

        self.trajectories_plot = []
        self.trajectories_scatter = []
        self.ellipses = []

    def generate_colors(self):
        nb_step = len(self.trajectories[0])
        return [c for c, _ in zip(cycle(SCANS_ID_TO_COLOR), range(nb_step))]

    def init(self):
        self.ax.clear()
        nb_step = len(self.trajectories[0])
        self.trajectories_plot = [self.ax.plot([], [], '-', c='black', alpha=0.5)[0] for _ in self.trajectories]
        self.trajectories_scatter = [self.ax.scatter([0] * nb_step, [0] * nb_step, c=self.generate_colors(), marker='o') for _ in self.trajectories]

        self.ellipses = []

        show_label = True
        for penalty in self.penalties:
            if isinstance(penalty, Penalty):
                self.ellipses += draw_elipse(self.ax, penalty.translation, penalty.cov, color='green',
                                         label='Penalties covariance' if show_label else None)
                show_label = False

        self.ax.set_title(self.label)
        self.ax.axis('equal')
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-5, 5)

    @property
    def static_lines(self):
        return self.ellipses

    @property
    def lines(self):
        return self.trajectories_plot + self.trajectories_scatter

    def update(self, i):
        # Each sample is a trajectory
        for trajectory, plot, scatter in zip(self.trajectories, self.trajectories_plot, self.trajectories_scatter):
            first_tf, _ = trajectory[0]
            x = [first_tf.x]
            y = [first_tf.y]
            c = self.generate_colors()
            # A trajectory is made of step (one step per pair of scan), which content iterations
            for _, step in enumerate(trajectory[1:]):
                final_tf, iters = step
                max_iter_sample = len(iters)
                iter = iters[min(max_iter_sample - 1, i)]  # A registration might end before the i-th iteration
                tf = iter["tf"]
                xi = tf[0, 2]
                yi = tf[1, 2]
                th = acos(tf[0, 0])
                x.append(xi)
                y.append(yi)
                # raw_arrow = Arrow(xi, yi, cos(th), sin(th), width=0.1, color='black')
                # if len(arrow) <= j:
                #     arrow.append(ax.add_patch(raw_arrow))
                # else:
                #     arrow[j].remove()
                #     arrow[j] = ax.add_patch(raw_arrow)
            # print(i, list(zip(x, y)))
            plot.set_data(x, y)
            scatter.set_offsets(list(zip(x, y)))
        self.ax.axis('equal')

class AnimatedRegistration:

    def __init__(self, title, experiments, scans, gt_poses, walls, pairwise_dumps):
        self.title = title
        self.experiments = experiments
        self.scans = [apply_tf_on_point_cloud(scan, pose) for scan, pose in zip(scans, gt_poses)]
        self.gt_poses = gt_poses
        self.walls = walls
        self.dump = pairwise_dumps

        """ experiments = { label => (trajectories, penalities), ...}
            trajectories = [(so2_tf, iter_datum), ...]
            penalities = [penalty, ...]
        """
        self.max_iter = max([len(iters) for trajectories, penalties in experiments.values() for trajectory in trajectories for final_tf, iters in trajectory])

        self.fig = plt.figure(figsize= (5* len(experiments), 6))

        self.exp_plots = [ExperimentationPlot(label, self.fig.add_subplot(1, len(experiments), i + 1), trajectories, penalties) for i, (label, (trajectories, penalties)) in enumerate(experiments.items())]
        # self.axis = [(label, self.fig.add_subplot(1, len(experiments), i + 1), exp[0], exp[1]) for i, (label, exp) in enumerate(experiments.items())]
        self.func_ani = None

    def init_animation(self):
        self.func_ani = FuncAnimation(self.fig, self.update, frames=range(0, self.max_iter), init_func=self.init, blit=False)

    def start_animation(self):
        plt.show()

    def plot_iter(self, i):
        self.init()
        self.update(i)
        plt.show()

    def plot_last_iter(self):
        self.plot_iter(self.max_iter)

    def init(self):
        # self.arrows = []
        self.lines = []
        self.static_lines = []
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85)
        for exp_plot in self.exp_plots:
            exp_plot.init()
            self.lines += exp_plot.lines
            self.static_lines += exp_plot.static_lines
            # The last step will have for reference a fusion of all scan except the last one, we then take the first iteration
            # descriptors_all_refs = self.dump[-1][0]["filtered_ref"]
            # descriptors_read = self.dump[0]["filtered_read"]
            # self.static_lines += self._plot_scan(ax, self.gt_poses[0].to_tf() @ descriptors_all_refs.numpy, color="black", descriptors=descriptors_all_refs)
            for id, (scan, gt_pose, color) in enumerate(zip(self.scans, self.gt_poses, cycle(SCANS_ID_TO_COLOR))):
                self.static_lines += self._plot_scan(exp_plot.ax, scan, gt_pose, f"Scan #{id}", color)
            # exp_plot.ax.legend(loc='upper left')
        return self.static_lines + self.lines

    def update(self, i):
        self.fig.suptitle(f"Iter {i:02} - {self.title} \n Distribution of registration in a simulated corridor ")
        for exp_plot in self.exp_plots:
            exp_plot.update(i)
        return self.static_lines + self.lines

    def save(self, filename):
        self.func_ani.save(filename, dpi=80, writer='imagemagick')

    def _plot_scan(self, axis, scan, origin=None, label="", color="blue", animated=False, descriptors=None):
        if origin is not None:
            lines = [axis.scatter(scan[0, :], scan[1, :], c=color, s=1, animated=animated)]  # , label=label+" scan points"
            lines += axis.plot(origin.x, origin.y, 'x', color=color, label=label+" origin" , animated=animated)
        else:
            lines = []

        if descriptors is not None:
            des_by_labels = descriptors.descriptors_by_labels
            if "normals" in des_by_labels:
                normals =  des_by_labels["normals"]
                for x, y, nx, ny in zip(scan[0, :], scan[1, :], normals[0, :], normals[1, :]):
                    arr = axis.arrow(x, y, nx/2, ny/2)
                    lines.append(arr)
            if "covariances" in des_by_labels:
                covs =  des_by_labels["covariances"]
                for i, (x, y, cov) in enumerate(zip(scan[0, :], scan[1, :], covs.transpose())):
                    lines += draw_elipse(axis, (x, y), cov.reshape(2, 2), color=color)
        return lines

    def _plot_map(self, ax, map):
        return [ax.plot((wall.p1.x, wall.p2.x), (wall.p1.y, wall.p2.y), 'black') for wall in map]
