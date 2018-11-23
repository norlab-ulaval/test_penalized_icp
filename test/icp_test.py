from pypm.icp import ICP, Penalty
from pypm.data_points import DataPoints
import numpy as np


A_3D__POINTCLOUD = DataPoints.from_list([[1, 1],
                                         [2, 2],
                                         [3, 3]])

A_2D_POINTCLOUD = DataPoints.from_list([[1, 1, 1, 1],
                                        [2, 2, 2, 2]])


def test_given_the_same_3d_point_as_ref_and_read_when_compute_returns_a_4x4_identity_matrix():
    icp = ICP()
    icp.load_from_dict(ICP.BASIC_CONFIG)
    res = icp.compute(A_3D__POINTCLOUD, A_3D__POINTCLOUD)
    assert (np.identity(4) == res).all()


def test_given_the_same_2d_point_as_ref_and_read_when_compute_returns_a_3x3_identity_matrix():
    icp = ICP()
    icp.load_from_dict(ICP.BASIC_CONFIG)
    res = icp.compute(A_2D_POINTCLOUD, A_2D_POINTCLOUD, np.identity(3))
    assert (np.identity(3) == res).all()


def test_given_a_penalty_with_config_that_ignore_penalty_the_res_is_not_affected():
    icp = ICP()
    icp.load_from_dict(ICP.BASIC_CONFIG)
    A_PENALTY = Penalty.from_translation(np.array([1, 2, 3]), np.identity(3))

    res = icp.compute(A_2D_POINTCLOUD, A_2D_POINTCLOUD, np.identity(3), [A_PENALTY])

    assert (np.identity(3) == res).all()

