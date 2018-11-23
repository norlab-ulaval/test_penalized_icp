import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

from picp.icp_sampling import icp_with_random_perturbation
from picp.scan_generator import ScanGenerator
from picp.util.geometry import generate_tf_mat
from picp.util.pose import Pose
from pypm.icp import ICP, Penalty


def plot_scan(axis, scan, origin, label="", color="blue", animated=False):
    lines = [axis.scatter(scan[0, :], scan[1, :], c=color, s=1, label=label+" scan points", animated=animated)]
    lines += axis.plot(origin.x, origin.y, 'x', color=color, label=label+" origin", animated=animated)
    return lines
    # for pt in scan_raw:
    #     axis.plot([pt[0], origin.x], [pt[1], origin.y], 'r--')


class AnimatedRegistration:

    def __init__(self, experiments, ref, read, origin, move, σ):
        self.experiments = experiments
        self.ref = ref
        self.read = read
        self.origin = origin
        self.move = move
        self.σ = σ
        self.max_iter = max([len(sample) for exp in experiments.values() for sample in exp])

        self.fig = plt.figure()
        self.axis = [self.fig.add_subplot(1, len(experiments), i + 1) for i, exp in enumerate(experiments)]
        self.func_ani = FuncAnimation(self.fig, self.update, frames=range(0, self.max_iter), init_func=self.init, blit=False)

    def init(self):
        self.lines = []
        self.static_lines = []
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.80)
        for ax in self.axis:
            ax.clear()
            self.lines += ax.plot([], [], '.', c='black', label="registration samples")
            self.static_lines += plot_scan(ax, self.ref, self.origin, "Reference", "blue")
            self.static_lines += plot_scan(ax, self.read, self.move, "Read", "red")
            ax.axis('equal')
            ax.legend(loc='lower right')

        return self.lines + self.static_lines

    def update(self, i):
        self.fig.suptitle(f"Iter {i:02} - Distribution of registration in a simulated corridor\n "
                          f"with a perturbation of {self.σ}")
        for (label, exp), ax, line in zip(self.experiments.items(), self.axis, self.lines):
            x = []
            y = []
            for j, sample_data in enumerate(exp):
                max_iter_sample = len(sample_data)
                iter = sample_data[min(max_iter_sample-1, i)] # A registration might end before the i-th iteration
                tf = iter["tf"]
                x.append(tf[0, 2])
                y.append(tf[1, 2])
            line.set_data(x, y)
            ax.axis('equal')
            ax.set_title(label)
        return self.lines + self.static_lines

    def save(self, filename):
        self.func_ani.save(filename, dpi=80, writer='imagemagick')

def icp_config():
    conf = icp.BASIC_CONFIG
    sensor_noise = [
        {"SimpleSensorNoiseDataPointsFilter": {"sensorType": 0}} # For LMS-150
    ]
    conf["readingDataPointsFilters"] = sensor_noise
    conf["referenceDataPointsFilters"] = sensor_noise
    conf["outlierFilters"] = [{"SensorNoiseOutlierFilter": {}}]
    conf["errorMinimizer"] = {"PointToPointWithPenaltiesErrorMinimizer": {"confidenceInPenalties": 0.9}}
    return conf

if __name__ == "__main__":
    sg = ScanGenerator()
    origin = Pose()
    move = Pose.from_values(0, 3, 0)
    ref = sg.generate(origin).transpose()
    read = sg.generate(move).transpose()

    init_tf = generate_tf_mat(move.position.array, angle=0)

    icp = ICP()
    #icp.set_default()
    icp.load_from_dict(ICP.BASIC_CONFIG)
    σ = 0.5
    cov = np.diag([σ**2, σ**2, 0.0])

    start = time.time()
    _, dataA = icp_with_random_perturbation(icp, read, ref, init_tf, nb_sample=100, cov=cov)
    print(f"Icp took {time.time()-start}s")

    icp.load_from_dict(icp_config())

    start = time.time()
    penalty = Penalty.from_translation(move.to_array()[0:2], np.array([[0.001, 0.00],
                                                                       [  0, 1]]))
    _, dataB = icp_with_random_perturbation(icp, read, ref, init_tf, nb_sample=100, cov=cov, penalties=[penalty])
    print(f"Icp took {time.time()-start:0.2f}s")

    experiments = OrderedDict({"Without": dataA, "With penalty": dataB})
    plotter = AnimatedRegistration(experiments, ref, read, origin, move, σ)
    #plotter.save("reg.gif")
    plt.show()


