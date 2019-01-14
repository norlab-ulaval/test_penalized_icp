import time
from collections import OrderedDict

import numpy as np

from picp.icp_sampling import icp_with_random_perturbation, icp_mapping_with_random_perturbation, icp_mapping
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


def icp_p_to_gaussian():
    conf = icp_covariance(knn=10)
    conf["errorMinimizer"] = {"PointToGaussianErrorMinimizer": {}}
    return conf

art = """XXXXX
XXXXX
XX...
XX.XX
XX.XX
*..XX
XXXXX"""

art = """XXXXXXX
XXXXXXX
X3...2X
X.XXX.X
X.XX1.X
.4*..XX
XXXXXXX"""

if __name__ == "__main__":
    origin = Pose()

    # poses = [origin,
    #          Pose(Position( 1, 0), 0),
    #          Pose(Position( 2, 0), 0),
    #          Pose(Position( 4, 0), 0),
    #          Pose(Position( 5, 0), 0),
    #          Pose(Position( 6, 0), 0),
    #
    #          Pose(Position( 6, 1), np.deg2rad(90)),
    #          Pose(Position( 6, 4), np.deg2rad(90)),
    #          Pose(Position( 6, 8), np.deg2rad(90)),
    #          ]

    walls, poses = from_ascii_art(art)
    poses[0] = origin
    sg = ScanGenerator(walls, nb_beam=180)
    print("Generating map...", end='', flush=True)
    scans = [sg.generate(p).transpose() for p in poses]
    print("done!")
    # init_tf = move.to_tf()

    icp = ICP()
    #icp.set_default()
    icp.load_from_dict(ICP.BASIC_CONFIG)
    no_penalties = [[] for p in poses]
    penalties_x = [from_cov_pose_to_penalties(p, np.array([[1, 0.00],
                                                           [0, 1e-5]])) for p in poses]
    penalties_y = [from_cov_pose_to_penalties(p, np.array([[1e-5, 0.00],
                                                           [0, 1]])) for p in poses]

    penalties_xy = [from_cov_pose_to_penalties(p, np.array([[1e-2, 0.00],
                                                            [0, 1e-2]])) for p in poses]
    experiments = [
        ("P2Plane no penalty", icp_plane_basic_config(), no_penalties),
        # ("P2Plane with penalty in y", icp_plane_penalties_config(), penalties_y),
        # ("With penalty in x", icp_p_to_gaussian(), penalties_x),
        ("P2Gaussian with penalty in y", icp_p_to_gaussian(), penalties_y),
        ("With penalty in xy", icp_p_to_gaussian(), penalties_xy),
    ]

    σ = 0.1
    σ_rot = np.deg2rad(0)
    pertur_cov = np.diag([σ**2, σ**2, σ_rot**2])
    experiments_res = OrderedDict()
    for label, config, penalties in experiments:
        start = time.time()
        icp.load_from_dict(config)
        icp.enable_dump(tf=True)
        trajectories = icp_mapping_with_random_perturbation(icp, scans, gt_tfs=poses, nb_sample=1, pertur_cov=pertur_cov, penalties=penalties)
        experiments_res[label] = (trajectories, penalties)
        print(f"The test '{label}' tooks {time.time()-start:0.3f}s")

    # To extract the sensor noise I do one registration with a dump of the descriptors
    icp.load_from_dict(icp_covariance())
    icp.enable_dump(tf=True, filtered_ref=True, filtered_read=True)
    steps = icp_mapping(icp, scans, gt_tfs=poses, update_map_with_gt=True)
    pairwise_dump = [s[1] for s in steps]  # dump for each pair of scans

    #, title, experiments, scans, gt_poses, walls, pairwise_dumps
    plotter = AnimatedRegistration("Test PointToGaussian", experiments_res, scans, poses, walls, pairwise_dump)
    plotter.init_animation()
    # plotter.plot_iter(0)
    # plotter.plot_last_iter()

    if False:
        print("Saving to gif...")
        plotter.save("reg.gif")
    plotter.start_animation()


