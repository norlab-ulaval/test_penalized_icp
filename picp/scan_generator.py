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
        σ = σ_d if dist < 1.646 else σ_r

        noisy_dist = σ * np.random.randn() + dist
        return origin + normalize(v) * noisy_dist


class Wall:
    def __init__(self, p1: Position, p2: Position):
        self.p1 = p1
        self.p2 = p2

class ScanGenerator:
    def __init__(self, sensor_model: SensorModel=LMS151()):
        self.walls = []
        self.nb_beam = 90
        self.max_range_beam = 100
        self.sensor_model = sensor_model

        self.add_wall(Wall(Position(-1, 6), Position(-1, -6)))
        self.add_wall(Wall(Position(+1, 6), Position(+1, -6)))

    def add_wall(self, wall: Wall):
        self.walls.append(wall)

    def generate(self, pose: Pose):
        origin = pose.position
        pts = []
        for beam_id in range(0, self.nb_beam):
            angle_rad = beam_id / self.nb_beam * 2 * math.pi
            end_beam = origin + Position.from_angle(angle_rad, norm=self.max_range_beam)
            closest_inter = None
            for wall in self.walls:
                inter = intersection_between_segments(origin, end_beam, wall.p1, wall.p2)
                if inter is None:
                    continue
                if closest_inter is None or (closest_inter - origin).norm > (inter - origin).norm:
                    closest_inter = inter

            if closest_inter is not None:
                point  = self.sensor_model.apply_noise(origin, closest_inter)
                pts.append(point.to_tuple())
        return np.array(pts)

