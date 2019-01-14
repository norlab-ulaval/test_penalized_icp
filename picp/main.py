import time
from collections import OrderedDict

import numpy as np

from picp.icp_sampling import icp_with_random_perturbation
from picp.reg_anim_plot import AnimatedRegistration
from picp.simulator.maps import create_basic_hallway, from_ascii_art
from picp.simulator.scan_generator import ScanGenerator
from picp.util.geometry import generate_rot_mat
from picp.util.pose import Pose
from pypm.icp import ICP, Penalty

from picp.util.position import Position


def icp_config():
    conf = ICP.BASIC_CONFIG.copy()
    conf['transformationCheckers'][0]['CounterTransformationChecker']['maxIterationCount'] = 40
    sensor_noise = [
        {"SimpleSensorNoiseDataPointsFilter": {"sensorType": 0}} # For LMS-150
    ]
    conf["readingDataPointsFilters"] = sensor_noise
    conf["referenceDataPointsFilters"] = sensor_noise
    conf["outlierFilters"] = [{"SensorNoiseOutlierFilter": {}}]
    conf["errorMinimizer"] = {"PointToPointWithPenaltiesErrorMinimizer": {"confidenceInPenalties": 0.5}}
    return conf

def icp_plane_basic_config():
    conf = ICP.BASIC_CONFIG.copy()

    conf["referenceDataPointsFilters"] = [
        {"SurfaceNormalDataPointsFilter": {"knn": 5}}
    ]
    conf["errorMinimizer"] = {"PointToPlaneErrorMinimizer": {}}
    return conf

def icp_plane_with_sensor_weight_config():
    conf = icp_plane_basic_config()

    sensor_noise = [
        {"SimpleSensorNoiseDataPointsFilter": {"sensorType": 0}} # For LMS-150
    ]
    conf["readingDataPointsFilters"] = sensor_noise
    conf["referenceDataPointsFilters"] += sensor_noise
    return conf

def icp_plane_penalties_config():
    conf = icp_config()
    conf["referenceDataPointsFilters"].append({"SurfaceNormalDataPointsFilter": {"knn": 5}})
    conf["errorMinimizer"] = {"PointToPlaneWithPenaltiesErrorMinimizer": {}}
    return conf

def icp_covariance(knn=5):
    conf = ICP.BASIC_CONFIG.copy()
    conf['transformationCheckers'][0]['CounterTransformationChecker']['maxIterationCount'] = 40
    sensor_noise = lambda cov: [
        {"SimpleSensorNoiseDataPointsFilter": {"sensorType": 0,  # For LMS-150
                                               "covariance": cov}}
    ]
    discretisation_est = [{"SurfaceCovarianceDataPointsFilter": {"knn": knn}},
                          {"DecomposeCovarianceDataPointsFilter": {"keepNormals": 0}}
                          ]
    conf["readingDataPointsFilters"] = sensor_noise(0)
    conf["referenceDataPointsFilters"] = sensor_noise(1) + discretisation_est
    conf["outlierFilters"] = [{"SensorNoiseOutlierFilter": {}}]
    conf["errorMinimizer"] = {"PointToPointWithPenaltiesErrorMinimizer": {"confidenceInPenalties": 0.5}}
    return conf


def icp_p_to_gaussian():
    conf = icp_covariance(knn=10)
    conf["errorMinimizer"] = {"PointToGaussianErrorMinimizer": {}}
    return conf


if __name__ == "__main__":
    origin = Pose()
    orientation = np.deg2rad(55)
    move = Pose(Position(0, 2).rotate(orientation), 0)

    walls = create_basic_hallway(orientation)

    sg = ScanGenerator(walls, nb_beam=180)
    ref = sg.generate(origin).transpose()
    read = sg.generate(move).transpose()

    init_tf = move.to_tf()

    icp = ICP()
    #icp.set_default()
    icp.load_from_dict(ICP.BASIC_CONFIG)
    σ = 0.5
    cov = np.diag([σ**2, σ**2, 0.0])

    #avg_penalty = Position.from_list([0, 2]).array
    # avg_penalty = move.to_array()[0:2]
    avg_penalty = move.position.rotate(orientation).array
    cov_penalty_x = Penalty(Pose(move.position).to_tf(), np.array([[1, 0.00],
                                                      [0, 1e-5]]))
    cov_penalty_y = Penalty(Pose(move.position).to_tf(), np.array([[1e-5, 0.00],
                                                      [    0, 1]]))
    cov_penalty_xy = Penalty(move.to_tf(), np.array([[0.5, 0.00],
                                                       [    0, 0.5]]))

    rot = generate_rot_mat(angle=orientation)
    cov_penalty_diag_x = cov_penalty_x
    cov_penalty_diag_x.cov = rot @ cov_penalty_diag_x.cov @ rot.transpose()
    cov_penalty_diag_y = cov_penalty_y
    cov_penalty_diag_y.cov = rot @ cov_penalty_diag_y.cov @ rot.transpose()
    # experiments = [("Without penalty",            ICP.BASIC_CONFIG, []),
    #                ("With penalty in x",  icp_config(), [cov_penalty_x]),
    #                ("With penalty in y",  icp_config(), [cov_penalty_y]),
    #                ("With penalty in xy", icp_config(), [cov_penalty_xy])
    #                ]
    # experiments = [("Without penalty", icp_plane_basic_config(), []),
                   # ("Without penalty, with sensor noise weight", icp_plane_with_sensor_weight_config(), []),
                   # ("With penalty in x", icp_plane_penalties_config(), [cov_penalty_x]),
                   # ("With penalty in y",  icp_plane_penalties_config(), [cov_penalty_y]),
                   # ("With penalty in xy", icp_plane_penalties_config(), [cov_penalty_xy])
                   # ]
    experiments = [
                   # ("Without penalty, p2p", ICP.BASIC_CONFIG, []),
                   # ("Without penalty, p2plane", icp_plane_basic_config(), []),
                   # ("Without penalty, p2Gaussian", icp_p_to_gaussian(), []),
                   ("With penalty in x", icp_p_to_gaussian(), [cov_penalty_diag_x]),
                   # ("With penalty in y",  icp_p_to_gaussian(), [cov_penalty_diag_y]),
                   # ("With penalty in xy", icp_p_to_gaussian(), [cov_penalty_xy])
                   ]

    experiments_res = OrderedDict()
    for label, config, penalties in experiments:
        print(label, "shoud have", len(penalties), "penalties")
        start = time.time()
        icp.load_from_dict(config)
        _, data = icp_with_random_perturbation(icp, read, ref, init_tf, nb_sample=100, cov=cov, penalties=penalties)
        experiments_res[label] = (data, penalties)
        print(f"The test '{label}' tooks {time.time()-start:0.3f}s")

    # To extract the sensor noise I do one registration with a dump of the descriptors
    icp.load_from_dict(icp_covariance())
    icp.enable_dump(tf=True, filtered_ref=True, filtered_read=True)
    _, dump = icp.compute(read, ref, init_tf)

    plotter = AnimatedRegistration("Test PointToGaussian", experiments_res, ref, read, origin, move, walls, σ, dump)
    # plotter.init_animation()
    plotter.plot_last_iter()

    if False:
        print("Saving to gif...")
        plotter.save("reg.gif")
    # plotter.start_animation()


