from pypm.icp import ICP
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
    res = icp.compute(A_2D_POINTCLOUD, A_2D_POINTCLOUD)
    assert (np.identity(3) == res).all()

