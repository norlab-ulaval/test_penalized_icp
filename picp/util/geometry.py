from typing import Optional

import numpy as np

from picp.util.pose import Pose
from picp.util.position import Position
from python.pypm.icp import Penalty


def from_cov_pose_to_penalties(pose, cov):
    penalty = Penalty(Pose(pose.position).to_tf(), cov)

    rot = generate_rot_mat(angle=pose.orientation)
    penalty.cov = rot @ penalty.cov @ rot.transpose()
    return penalty


def generate_rot_mat(angle: float=0):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), +np.cos(angle)]])


def generate_tf_mat(trans: np.ndarray=np.array([0, 0]), angle: float=0):
    return np.array([[np.cos(angle), -np.sin(angle), trans[0]],
                     [np.sin(angle), +np.cos(angle), trans[1]],
                     [            0,              0,       1]])

def extract_xyt_from_tf_mat(tf: np.ndarray):
    return (tf[0, 2],  # x
            tf[1, 2],  # y
            np.arccos(tf[0, 0]) if 1.0 - tf[0, 0] > 1e-6 else 0.0)  # theta

def make_homogeneous(pts):
    ones = np.ones((1, pts.shape[1]))
    return np.vstack((pts, ones))

def apply_tf_on_point_cloud(pts: np.ndarray, tf: [Pose, np.ndarray]):
    if isinstance(tf, Pose):
        tf = tf.to_tf()
    pts_moved = tf @ np.vstack([pts, np.ones((1, pts.shape[1]))])
    return pts_moved[0:2, :]

def normalize(vec: Position) -> Position:
    if vec.norm == 0:
        raise ZeroDivisionError
    return vec.copy() / vec.norm

def projection(reference: Position, start: Position, end: Position) -> float:
    start_to_end = normalize(end - start)
    start_to_reference = reference - start
    return np.inner(start_to_reference.array, start_to_end.array).view(float)


def closest_point_on_line(reference: Position, start: Position, end: Position) -> Position:
    return start + normalize(end - start) * projection(reference, start=start, end=end)


def closest_point_on_segment(reference: Position, start: Position, end: Position) -> Position:
    if end == start:
        return start
    proj = projection(reference, start=start, end=end)
    if proj >= (end - start).norm:
        return end
    elif proj <= 0:
        return start
    else:
        return closest_point_on_line(reference, start=start, end=end)

def intersection_between_segments(a1: Position, a2: Position, b1: Position, b2: Position) -> Optional[Position]:
    angle_1 = (b1-a1).angle
    angle_2 = (b2-a1).angle
    target_angle = (a2-a1).angle
    if target_angle >= max(angle_1, angle_2) or min(angle_1, angle_2) >= target_angle:
        return None
    try:
        inter = intersection_between_lines(a1, a2, b1, b2)
    except ValueError:
        return None

    # Check that the intersection point lie inside the segment
    if inter == closest_point_on_segment(inter, a1, a2) == closest_point_on_segment(inter, b1, b2):
        return inter
    return None


def intersection_between_line_and_segment(seg1: Position, seg2: Position, line1: Position, line2: Position) -> Optional[Position]:
    try:
        inter = intersection_between_lines(seg1, seg2, line1, line2)
    except ValueError:
        return None

    if inter == closest_point_on_segment(inter, seg1, seg2):
        return inter
    return None


def intersection_between_lines(a1: Position, a2: Position, b1: Position, b2: Position) -> Position:
    s = np.vstack([a1.array, a2.array, b1.array, b2.array])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])  # first line
    l2 = np.cross(h[2], h[3])  # second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:
        raise ValueError('Parallel lines')
    return Position(x / z, y / z)