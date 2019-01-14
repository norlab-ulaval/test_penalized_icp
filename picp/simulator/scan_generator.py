import math

import numpy as np

from picp.util.geometry import intersection_between_segments, normalize
from picp.util.pose import Pose
from picp.util.position import Position

class SensorModel:
    def apply_noise(self, origin: Position, intersection: Position):
        raise NotImplementedError("Abstract function.")

class LMS151(SensorModel):
    def apply_noise(self, origin: Position, intersection: Position):
        v = intersection - origin
        dist = v.norm
        # Noise modeled based on "Noise characterization of depth sensors for surface inspections", 2015
        σ_r = (6.8 * dist + 0.81) / 1000
        σ_d = 0.012
        if True:
            σ = σ_d if dist < 1.646 else σ_r
        else:
            σ = 0.0001

        noisy_dist = σ * np.random.randn() + dist
        if v.norm == 0.0:
            return origin
        return origin + normalize(v) * noisy_dist


class ScanGenerator:
    def __init__(self, walls, nb_beam = 180, sensor_model: SensorModel=LMS151()):
        self.walls = walls
        self.nb_beam = nb_beam
        self.max_range_beam = 100
        self.sensor_model = sensor_model

    def generate(self, pose: Pose):
        origin = pose.position
        orientation = pose.orientation + np.pi / 2  # The first point taken by the sensor is parallel to the y axis
        pts = []
        for beam_id in range(0, self.nb_beam):
            angle_rad = beam_id / self.nb_beam * 2 * math.pi + orientation
            end_beam = origin + Position.from_angle(angle_rad, norm=self.max_range_beam)
            closest_inter = None
            for wall in self.walls:
                inter = intersection_between_segments(origin, end_beam, wall.p1, wall.p2)
                if inter is None:
                    continue
                if closest_inter is None or (closest_inter - origin).norm > (inter - origin).norm:
                    closest_inter = inter

            if closest_inter is not None:
                point  = self.sensor_model.apply_noise(origin, closest_inter) - origin
                scan_frame_point = point.rotate(-pose.orientation)
                pts.append(scan_frame_point.to_tuple())

        return np.array(pts)

