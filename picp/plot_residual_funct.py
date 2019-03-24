import numpy as np
from pypm import ICP

from picp.main import icp_config, art
from picp.plot_trajectory import icp_p_to_gaussian
from picp.simulator.maps import create_basic_hallway, from_ascii_art
from picp.simulator.scan_generator import ScanGenerator
from picp.util.pose import Pose
import matplotlib.pyplot as plt


def icp_plane_penalties_config():
    conf = icp_config()
    conf["referenceDataPointsFilters"].append({"SurfaceNormalDataPointsFilter": {"knn": 5}})
    conf["errorMinimizer"] = {"PointToPlaneWithPenaltiesErrorMinimizer": {}}
    return conf

def icp_covariance(knn=5):
    conf = ICP.BASIC_CONFIG.copy()
    #conf['matcher']['KDTreeMatcher']['knn'] = knn
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


# def icp_p_to_gaussian(knn=5):
#     conf = icp_covariance(knn=knn)
#     conf["errorMinimizer"] = {"PointToGaussianErrorMinimizer": {}}
#     return conf


if __name__ == "__main__":
    # origin = Pose()
    # orientation = np.deg2rad(55)
    #
    # walls = create_basic_hallway(orientation)
    # sg = ScanGenerator(walls, nb_beam=180)
    # ref = sg.generate(origin).transpose()
    orientation = np.deg2rad(0)
    walls, poses = from_ascii_art(art, orientation=orientation)
    sg = ScanGenerator(walls, nb_beam=180)
    print("Generating map...")

    ref = sg.generate(poses[0], check_cache=True).transpose()

    # ref = np.array([[5, 5]])
    max_v = 20
    # experiments = [("P2Gaussian knn=10", icp_p_to_gaussian(knn=10), max_v),
    #                ("P2Gaussian knn=20", icp_p_to_gaussian(knn=20), max_v),
    #                ("P2Gaussian knn=30", icp_p_to_gaussian(knn=30), max_v)
    #               ]
    experiments = [
                    ("P2Gaussian knn=5", icp_p_to_gaussian(knn=5), max_v),
                    ("P2Gaussian knn=10", icp_p_to_gaussian(knn=10), max_v),
                    ("P2Gaussian knn=20", icp_p_to_gaussian(knn=20), max_v)
                  ]

    # experiments = [("P2Point", ICP.BASIC_CONFIG, 7),
    #                ("P2Plan", icp_plane_penalties_config(), 7),
    #                ("P2Gaussian", icp_p_to_gaussian(), 1000)]

    # np.set_printoptions(threshold=np.nan)
    # print(residuals)

    fig = plt.figure()

    icp = ICP()
    for i, (label, config, max_value) in enumerate(experiments):
        icp.load_from_dict(config)
        print(f"Computing `{label}` max_val:`{max_value}`")
        nb_sample = 300
        residuals = icp.compute_residual_function(ref, [-8, -3], [13, 4], nb_sample)
        # residuals = icp.compute_residual_function(ref, [-4, -4], [4, 4], 100)
        min_value = 0

        ax = fig.add_subplot(1, len(experiments), i + 1)
        ax.set_title(label)
        ax.axis('equal')
        residual = np.clip(residuals[:, 2], min_value, max_value)
        # residual = residuals[:, 2]
        hb = ax.hexbin(residuals[:, 0], residuals[:, 1], residual, gridsize=2 * nb_sample // 3, cmap='inferno')
        fig.colorbar(hb, ax=ax, extend='max')
    plt.show()




