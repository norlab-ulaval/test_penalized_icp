import hashlib
import math
import pickle

import numpy as np

from pathlib import Path

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

    def generate(self, pose: Pose, check_cache=False):
        if check_cache:
            if self.scan_exist(pose):
                return self.load_scan(pose)

        origin = pose.position
        orientation = pose.orientation + np.pi / 2  # The first point taken by the sensor is parallel to the y axis
        pts = []
        for beam_id in range(0, self.nb_beam):
            angle_rad = beam_id / self.nb_beam * 2 * math.pi + orientation
            end_beam = origin + Position.from_angle(angle_rad, norm=self.max_range_beam)
            closest_inter = None
            for wall in self.walls:
                # inter = intersection_between_ray_and_segment(origin, end_beam - origin, wall.p1, wall.p2)
                inter = intersection_between_segments(origin, end_beam, wall.p1, wall.p2)
                if inter is None:
                    continue
                if closest_inter is None or (closest_inter - origin).norm > (inter - origin).norm:
                    closest_inter = inter

            if closest_inter is not None:
                point  = self.sensor_model.apply_noise(origin, closest_inter) - origin
                scan_frame_point = point.rotate(-pose.orientation)
                pts.append(scan_frame_point.to_tuple())
        scan = np.array(pts)
        self.save_scan(pose, scan)
        return scan

    def scan_exist(self, pose):
        if not self._cache_root().exists():
            return False
        return self._path_to_cache(pose).exists()

    def load_scan(self, pose):
        path = self._path_to_cache(pose)
        print(f"Loading from cache {path}")
        return pickle.load(open(path, "rb"))

    def save_scan(self, pose, scan):
        folder = self._cache_root()
        if not folder.exists():
            folder.mkdir()
        path = self._path_to_cache(pose)
        print(f"Saving scan into the cache {path}")
        pickle.dump(scan, open(path, "wb+"))

    def _generate_hash(self, pose):
        value = str(pose) + "".join([str(w) for w in self.walls]) + str(self.nb_beam)
        return hashlib.md5(str(value).encode()).hexdigest()

    def _cache_root(self):
        return Path.home() / '.cache' / 'picp'

    def _path_to_cache(self, pose):
        return self._cache_root() / self._generate_hash(pose)




