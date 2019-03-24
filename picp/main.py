import time
from collections import OrderedDict

import numpy as np

from picp.icp_sampling import icp_with_random_perturbation, icp_mapping
from picp.plot_trajectory import icp_p_to_gaussian
from picp.reg_anim_plot import AnimatedRegistration
from picp.simulator.maps import create_basic_hallway, from_ascii_art
from picp.simulator.scan_generator import ScanGenerator
from picp.util.geometry import generate_rot_mat, from_cov_pose_to_penalties
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


def icp_p_to_gaussian_old():
    conf = icp_covariance(knn=10)
    conf["errorMinimizer"] = {"PointToGaussianErrorMinimizer": {}}
    return conf

art = """XXXXXXX
XXXXXXX
X3...2X
X.XXX.X
X4XX1.X
.5*..XX
XXXXXXX"""

if __name__ == "__main__":
    # origin = Pose()
    # orientation = np.deg2rad(0)
    # move = Pose(Position(0, 2).rotate(orientation), 0)
    #
    # walls = create_basic_hallway(orientation)
    #
    # sg = ScanGenerator(walls, nb_beam=180)
    # ref = sg.generate(origin).transpose()
    # read = sg.generate(move).transpose()

    orientation = np.deg2rad(0)
    walls, poses = from_ascii_art(art, orientation=orientation)
    # poses = [Pose.from_values(p.x, p.y, orientation) for p in poses]
    # poses[0] = origin
    sg = ScanGenerator(walls, nb_beam=180)
    print("Generating map...")

    scans = [sg.generate(p, check_cache=True).transpose() for p in poses]

    origin, ref = poses[0], scans[0]
    move, read = poses[1], scans[1]

    poses = [origin, move]
    scans = [ref, read]

    init_tf = move.to_tf()

    icp = ICP()
    #icp.set_default()
    icp.load_from_dict(ICP.BASIC_CONFIG)
    σ = 0.5
    cov = np.diag([σ**2, σ**2, 0.0])

    #avg_penalty = Position.from_list([0, 2]).array
    # avg_penalty = move.to_array()[0:2]
    avg_penalty = move.position.rotate(orientation).array
    cov_penalty_x = from_cov_pose_to_penalties(move, np.array([[1, 0.00],
                                                               [0, 1e-5]]))
    cov_penalty_y = from_cov_pose_to_penalties(move, np.array([[1e-5, 0.00],
                                                               [    0, 1]]))
    cov_penalty_xy = from_cov_pose_to_penalties(move, np.array([[0.5, 0.00],
                                                                [    0, 0.5]]))

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
                   ("With penalty in x", icp_p_to_gaussian(), [cov_penalty_x]),
                   # ("With penalty in y",  icp_p_to_gaussian(), [cov_penalty_diag_y]),
                   # ("With penalty in xy", icp_p_to_gaussian(), [cov_penalty_xy])
                   ]

    experiments_res = OrderedDict()
    for label, config, penalties in experiments:
        print(label, "shoud have", len(penalties), "penalties")
        start = time.time()
        icp.load_from_dict(config)
        so2_tfs, iter_data = icp_with_random_perturbation(icp, read, ref, init_tf, nb_sample=100, pertur_cov=cov, penalties=penalties)
        # data = [iter_datum]
        # trajectories = [(so2_tf, iter_datum), ...]
        trajectories = [[(Pose.from_values(*so2_tf), iter_datum)] for so2_tf, iter_datum in zip(so2_tfs, iter_data)]
        experiments_res[label] = (trajectories, penalties)
        print(f"The test '{label}' took {time.time()-start:0.3f}s")

    # To extract the sensor noise I do one registration with a dump of the descriptors
    icp.load_from_dict(icp_covariance())
    icp.enable_dump(tf=True, filtered_ref=True, filtered_read=True)
    steps = icp_mapping(icp, scans, gt_tfs=poses, update_map_with_gt=True)
    pairwise_dump = [s[1] for s in steps]

    plotter = AnimatedRegistration("Test PointToGaussian", experiments_res, scans, poses, walls, pairwise_dump)
    # plotter.init_animation()
    plotter.plot_last_iter()

    if False:
        print("Saving to gif...")
        plotter.save("reg.gif")
    # plotter.start_animation()
