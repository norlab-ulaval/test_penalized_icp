import math
from typing import List, Optional

import numpy as np
from pypm import DataPoints
from pypm.icp import ICP, Penalty

from picp.util.geometry import generate_tf_mat, extract_xyt_from_tf_mat, make_homogeneous
from picp.util.pose import Pose


def icp_with_random_perturbation(icp: ICP, read: DataPoints, ref: DataPoints, init_tf: np.ndarray,
                                 nb_sample: int, pertur_cov: np.ndarray, penalties: List[Penalty]=[]):
    tfs = []
    iter_data = []
    perturbations = np.random.multivariate_normal([0, 0, 0], pertur_cov, nb_sample)
    for i in range(0, nb_sample):
        pertu = generate_tf_mat(perturbations[i, 0:2], perturbations[i, 2])
        icp.enable_dump(tf=True)
        tf, iter_datum = icp.compute(read, ref, init_tf @ pertu, penalties)
        iter_data.append(iter_datum)
        tfs.append((tf[0, 2], #x
                    tf[1, 2], #y
                    np.arccos(tf[0, 0]) if 1.0 - tf[0, 0] > 1e-6 else 0.0)) # theta

    return np.array(tfs), iter_data


def icp_mapping_with_random_perturbation(icp: ICP, scans: List[DataPoints], gt_tfs: List[np.ndarray],
                                         nb_sample: int, pertur_cov: np.ndarray, penalties: List[Penalty]=[]):
    trajectories = []
    for _ in range(0, nb_sample):
        steps = icp_mapping(icp, scans, gt_tfs, pertur_cov, penalties)
        trajectories.append(steps)
    return trajectories


def apply_tf_on_scan(tf: np.ndarray, scan: np.ndarray):
    moved_scan = tf @ make_homogeneous(scan)
    return moved_scan[0:2, :]  # Remove homogeneous


def icp_mapping(icp: ICP, scans: List[np.ndarray], gt_tfs: List[Pose],  pertur_cov: np.ndarray=None, penalties: List[Penalty]=[],
                update_map_with_gt=False):
    steps = []
    if pertur_cov is None:
        perturbations = np.zeros((len(scans), 3))
    else:
        perturbations = np.random.multivariate_normal([0, 0, 0], pertur_cov, len(scans))

    # Convert global frame tf to between scan
    between_scan_tfs = [curr.to_tf() @ np.linalg.inv(prev.to_tf()) for prev, curr in zip(gt_tfs, gt_tfs[1:])]

    # The first scan is always the map
    map = apply_tf_on_scan(gt_tfs[0].to_tf(), scans[0])
    prev_tf = gt_tfs[0]
    steps.append((gt_tfs[0], {}))
    for i, (read, between_scan_tf, penalty) in enumerate(zip(scans[1:], between_scan_tfs, penalties[1:])):
        pertu = generate_tf_mat(perturbations[i, 0:2], perturbations[i, 2])

        # `prev_tf` is in the global frame, its the registration result of the previous pair of scans.
        init_tf = between_scan_tf @ prev_tf.to_tf() @ pertu
        #print(f"{i}: gt_tf{gt_tfs[i+1]}\n  init_tf{Pose.from_tf(init_tf)}, prev_tf{prev_tf} between_scan {between_scan_tf}")
        # Note: `penalty` are in global frame and are independent from the `init_tf`
        tf, iter_datum = icp.compute(read, map, init_tf, penalty if isinstance(penalty, list) else [penalty])

        so2_tf = Pose.from_tf(tf)
        # print(f"gt {extract_xyt_from_tf_mat(gt_tf.to_tf())} vs result {so2_tf}")
        steps.append((so2_tf, iter_datum))

        # This flag used the ground truth to assemble the map instead of the registration results
        if update_map_with_gt:
            moved_read = apply_tf_on_scan(gt_tfs[i+1].to_tf(), read)
            prev_tf = gt_tfs[i+1]
        else:
            moved_read = apply_tf_on_scan(tf, read)
            prev_tf = so2_tf
            # prev_tf.orientation = gt_tfs[i+1].orientation # Ignore rotation results
        map = np.hstack((map, moved_read))
    return steps