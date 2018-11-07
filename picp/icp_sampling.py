import math

import numpy as np
from pypm import DataPoints
from pypm.icp import ICP

from picp.util.geometry import generate_tf_mat


def icp_with_random_perturbation(icp: ICP, read: DataPoints, ref: DataPoints, init_tf: np.ndarray,
                                 nb_sample: int, cov: np.ndarray):
    tfs = []
    perturbations = np.random.multivariate_normal([0, 0, 0], cov, nb_sample)
    for i in range(0, nb_sample):
        pertu = generate_tf_mat(perturbations[i, 0:2], perturbations[i, 2])
        tf = icp.compute(read, ref, init_tf @ pertu)
        tfs.append((tf[0, 2], #x
                    tf[1, 2], #y
                    np.arccos(tf[0, 0]) )) # theta

    return np.array(tfs)