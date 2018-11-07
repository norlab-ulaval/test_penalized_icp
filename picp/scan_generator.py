import math

import numpy as np

from picp.util.geometry import intersection_between_segments
from picp.util.pose import Pose
from picp.util.position import Position


class Wall:
    def __init__(self, p1: Position, p2: Position):
        self.p1 = p1
        self.p2 = p2

class ScanGenerator:
    def __init__(self):
        self.walls = []
        self.nb_beam = 90
        self.max_range_beam = 100

        self.add_wall(Wall(Position(-1, 6), Position(-1, -6)))
        self.add_wall(Wall(Position(+1, 6), Position(+1, -6)))

    def add_wall(self, wall: Wall):
        self.walls.append(wall)

    def generate(self, pose: Pose):
        origin = pose.position
        pts = []
        for beam_id in range(0, self.nb_beam):
            angle_rad = beam_id * (360 / self.nb_beam) * math.pi / 180
            end_beam = origin + Position.from_angle(angle_rad, norm=self.max_range_beam)
            closest_inter = None
            for wall in self.walls:
                inter = intersection_between_segments(origin, end_beam, wall.p1, wall.p2)
                if inter is None:
                    continue
                if closest_inter is None or (closest_inter - origin).norm > (inter - origin).norm:
                    closest_inter = inter
            if closest_inter is not None:
                pts.append(closest_inter.to_tuple())
        return np.array(pts)

