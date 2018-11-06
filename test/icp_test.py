from pypm.icp import ICP
from pypm.data_points import DataPoints
import numpy as np


A_POINTCLOUD = DataPoints.from_list([[1, 1],
                                     [2, 2],
                                     [3, 3]])


def test_given_the_same_point_as_ref_and_read_when_compute_returns_a_identity_matrix():
    icp = ICP()
    icp.load_from_dict(ICP.BASIC_CONFIG)
    res = icp.compute(A_POINTCLOUD, A_POINTCLOUD)
    assert (np.identity(4) == res).all()

