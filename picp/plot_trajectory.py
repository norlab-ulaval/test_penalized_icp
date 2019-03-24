import time
from collections import OrderedDict

import numpy as np

from picp.config_builder import ConfigBuilder
from picp.icp_sampling import icp_with_random_perturbation, icp_mapping_with_random_perturbation, icp_mapping
from picp.reg_anim_plot import AnimatedRegistration
from picp.simulator.maps import create_basic_hallway, from_ascii_art
from picp.simulator.scan_generator import ScanGenerator
from picp.util.geometry import generate_rot_mat, from_cov_pose_to_penalties
from picp.util.pose import Pose
from pypm.icp import ICP, Penalty

from picp.util.position import Position


# def icp_covariance(knn=5):
#     conf = ICP.BASIC_CONFIG.copy()
#     conf['transformationCheckers'][0]['CounterTransformationChecker']['maxIterationCount'] = 40
#     sensor_noise = lambda cov: [
#         {"SimpleSensorNoiseDataPointsFilter": {"sensorType": 0,  # For LMS-150
#                                                "covariance": cov}}
#     ]
#     discretisation_est = [{"SurfaceCovarianceDataPointsFilter": {"knn": knn}},
#                           {"DecomposeCovarianceDataPointsFilter": {"keepNormals": 0}}
#                           ]
#     conf["readingDataPointsFilters"] = sensor_noise(0)
#     conf["referenceDataPointsFilters"] = sensor_noise(1) + discretisation_est
#     conf["outlierFilters"] = [{"SensorNoiseOutlierFilter": {}}]
#     conf["errorMinimizer"] = {"PointToPointWithPenaltiesErrorMinimizer": {"confidenceInPenalties": 0.5}}
#     return conf

def icp_p_to_gaussian(knn=5):
    base_builder = ConfigBuilder()#.add_outlier_filter_trim()
    p2gauss = base_builder.copy().with_point_to_gaussian() \
        .add_sensor_noise_to_read() \
        .add_sensor_noise_to_ref(generate_cov=True).add_cov_to_ref(knn).add_decompose_cov_to_ref()
    return p2gauss.build()

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
X4XX1.X
.5*..XX
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

    orientation = np.deg2rad(0)
    walls, poses = from_ascii_art(art, origin_offset=Position(1, 1), orientation=orientation)
    # poses = [Pose.from_values(p.x, p.y, orientation) for p in poses]
    # poses[0] = origin
    sg = ScanGenerator(walls, nb_beam=180)
    print("Generating map...")

    scans = [sg.generate(p, check_cache=True).transpose() for p in poses]
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
    penalties_xy3 = [from_cov_pose_to_penalties(p, np.array([[1e-3, 0.00],
                                                            [0, 1e-3]])) for p in poses]

    base_builder = ConfigBuilder().add_outlier_filter_trim(overlap=0.95)
    p2plane = base_builder.copy().with_point_to_plane().add_normal_to_ref()
    p2gauss = base_builder.copy().with_point_to_gaussian()\
        .add_sensor_noise_to_read()\
        .add_sensor_noise_to_ref(generate_cov=True).add_cov_to_ref().add_decompose_cov_to_ref()

    # For ploting the final map
    p2p_with_cov = base_builder.copy().with_point_to_point()\
        .add_sensor_noise_to_read()\
        .add_sensor_noise_to_ref(generate_cov=True).add_cov_to_ref().add_decompose_cov_to_ref()

    experiments = [
        # ("With penalty in x", icp_p_to_gaussian(), penalties_x),
        # ("P2Plane with penalty in y", icp_plane_penalties_config(), penalties_x),
        # ("P2Gaussian with penalty in y", icp_p_to_gaussian(), penalties_y),
        ("P2Plane no penalty", p2plane.build(), no_penalties),
        ("P2Plan with penalty in xy, $\sigma = 10^{-2}$", p2plane.build(), penalties_xy),
        ("P2Gaussian no penalty", p2gauss.build(), no_penalties),
        ("P2Gaussian with penalty in xy, $\sigma = 10^{-2}$", p2gauss.build(), penalties_xy),
        # ("With penalty in xy, $\sigma = 10^{-3}$", icp_p_to_gaussian(), penalties_xy3),
    ]

    σ = 0.3
    σ_rot = np.deg2rad(3)
    pertur_cov = np.diag([σ**2, σ**2, σ_rot**2])
    experiments_res = OrderedDict()
    for label, config, penalties in experiments:
        start = time.time()
        icp.load_from_dict(config)
        icp.enable_dump(tf=True)
        trajectories = icp_mapping_with_random_perturbation(icp, scans, gt_tfs=poses, nb_sample=15, pertur_cov=pertur_cov, penalties=penalties)
        experiments_res[label] = (trajectories, penalties)
        print(f"The test '{label}' took {time.time()-start:0.3f}s")

    # To extract the sensor noise I do one registration with a dump of the descriptors
    icp.load_from_dict(p2p_with_cov.build())
    icp.enable_dump(tf=True, filtered_ref=True, filtered_read=True)
    steps = icp_mapping(icp, scans, gt_tfs=poses, update_map_with_gt=True)
    pairwise_dump = [s[1] for s in steps]  # dump for each pair of scans

    #, title, experiments, scans, gt_poses, walls, pairwise_dumps
    plotter = AnimatedRegistration("Test PointToGaussian", experiments_res, scans, poses, walls, pairwise_dump)
    plotter.init_animation()
    #plotter.plot_iter(0)
    # plotter.plot_last_iter()

    if False:
        print("Saving to gif...", end='', flush=True)
        plotter.save("reg.gif")
        print(" done!")
    plotter.start_animation()


