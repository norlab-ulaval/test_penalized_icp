from typing import List, Dict, Tuple

import numpy as np
import pypm.pypm_core as core


class DataPoints:
    """ Wrapper around libpointmatcher's DataPoints class
    Under the hood, the points are represented using an Eigen Matrix"""

    def __init__(self, raw_cpp_dp=None):
        # TODO: Find a way to make the constructor private
        self.raw_cpp_dp = raw_cpp_dp

    @property
    def numpy(self):
        return self.raw_cpp_dp.get_features()

    @property
    def descriptors(self):
        return self.raw_cpp_dp.get_descriptors()

    @property
    def descriptors_by_labels(self) -> Dict[str, np.ndarray]:
        des_by_label = {}
        descriptors = self.descriptors
        index = 0
        for text, span in self.descriptor_labels:
            des_by_label[text] = descriptors[index:index+span, :]
            index += span
        return des_by_label

    @property
    def descriptor_labels(self) -> List[Tuple[str, int]]:
        return self.raw_cpp_dp.get_descriptor_labels()

    @property
    def shape(self):
        return self.raw_cpp_dp.get_shape()

    @classmethod
    def from_list(cls, m: List[List[float]], make_homogeneous=True):
        return cls.from_numpy(np.array(m, dtype=np.float64), make_homogeneous=make_homogeneous)

    @classmethod
    def from_numpy(cls, m: np.ndarray, make_homogeneous=True):
        """Convert a numpy array to a DataPoints, it also convert it to homogeneous.
        The vectors must be formatted in a column wise format:
        [[x1,x2, ..],
         [y1,y2, ..],
         [z1,z2, ..]]"""
        if m.shape[0] not in [2, 3] and 0 < m.shape[1]:
            raise RuntimeError("The DataPoints must have a columns wise vector. With a shape (2, N) or (3, N).")

        if m.dtype != np.float64:
            m = m.astype(np.float64)

        if make_homogeneous:
            ones = np.ones((1, m.shape[1]))
            m = np.vstack((m, ones))
        return DataPoints(core.from_ndarray_to_datapoint(m))