from typing import Dict, Optional, Tuple, List

import numpy as np
import yaml

import pypm.pypm_core as core
from pypm.data_points import DataPoints

class Penalty:
    def __init__(self, tf: np.ndarray, cov: np.ndarray):
        assert tf.shape[0] == tf.shape[1], "Transformation matrix must be a squared matrix"
        assert cov.shape[0] == cov.shape[1], "Covariance matrix must be a squared matrix"
        assert tf.shape[0] -1  == cov.shape[0], "Transformation matrix must have be (N+1) by (N+1) and Covariance NxN"

        self.tf = tf
        self.cov = cov

    @classmethod
    def from_translation(cls, trans_vec: np.ndarray, cov: np.ndarray):
        """
        Create penalty from a translation vector
        :param trans_vec: Nx1 vector
        :param cov: NxN matrix
        """
        N = trans_vec.shape[0]
        tf = np.identity(N + 1)
        tf[0:N, N] = trans_vec
        return cls(tf, cov)

    @property
    def translation(self) -> np.ndarray:
        d = self.tf.shape[0]
        return self.tf[0:d-1, d-1]

    def to_tuple(self):
        return self.tf, self.cov


class ICP:
    """ Wrapper around libpointmatcher's Iterative Closest Point class"""

    BASIC_CONFIG = {
        'outlierFilters': [],
        'errorMinimizer': 'PointToPointErrorMinimizer',
        'matcher': {
            'KDTreeMatcher': {
                'knn': 1
            }
        },
        'transformationCheckers': [
            {'CounterTransformationChecker': {'maxIterationCount': 40}},
            'DifferentialTransformationChecker'
        ],
        'inspector': 'NullInspector'
    }

    def __init__(self):
        self.icp = core.ICP()
        self._is_dump_enable = False
        self.dump_config = {}

    def set_default(self):
        """Configure ICP with the defaults from libpointmatcher"""
        self.icp.set_default()

    def load_from_dict(self, d: Dict):
        """Load a configuration from a dict"""
        self.load_from_yaml(yaml.dump(d))

    def load_from_yaml_file(self, path: str):
        with open(path) as file:
            self.load_from_yaml(file.read())

    def load_from_yaml(self, yaml_str: str):
        """ Load a configuration from a string in a yaml format"""
        self.icp.load_from_yaml(yaml_str)

    def enable_dump_all(self):
        self.enable_dump(tf=True,
                         descriptors=True,
                         matches=True,
                         outlier_weight=True,
                         filtered_ref=True,
                         filtered_read=True)

    def enable_dump(self, *, # Force the keyword argument
                    tf: bool=True,
                    matches: bool=False,
                    outlier_weight: bool=False,
                    filtered_ref: bool=False,
                    filtered_read: bool=False):
        dict_local = dict(locals())
        args = ["tf", "matches", "outlier_weight", "filtered_ref", "filtered_read"]
        self.dump_config = {arg: dict_local[arg] for arg in args if arg in dict_local and dict_local[arg]}
        self._is_dump_enable = True

    def disable_dump(self):
        self._is_dump_enable = False

    def compute_residual_function(self, reference: [np.ndarray, DataPoints],
                                  mins: [list],
                                  maxs: [list],
                                  p_nb_samples_per_dim: [list] = 100) -> np.ndarray:
        if isinstance(reference, np.ndarray):
            reference = DataPoints.from_numpy(reference)

        dim_ref = reference.shape[0] -1
        assert dim_ref == len(mins) == len(maxs), "The mins and maxs must have same number of dimensions as the reference"

        if isinstance(p_nb_samples_per_dim, int):
            nb_samples_per_dim = [p_nb_samples_per_dim for _ in range(dim_ref)]
        else:
            nb_samples_per_dim = p_nb_samples_per_dim
        return self.icp.compute_residual_function(reference.raw_cpp_dp,
                                                  mins,
                                                  maxs,
                                                  nb_samples_per_dim)


    def compute(self,
                read: [np.ndarray, DataPoints],
                reference: [np.ndarray, DataPoints],
                init_tf: Optional[np.ndarray]=None,
                penalties: List[Penalty]=[]) -> Tuple[np.ndarray, List]:
        """
        Register the 'reading' point cloud against the 'reference'.
        In the context of mapping reference is the map and the reading is the scan.

        The point clouds are expected to have a 3xN or 2xN format, where N is the number of points.
        :param read: Reading point cloud, the points that will be move.
        :param reference: Reference point cloud.
        :param init_tf: Initial transformation apply to the reading. The performance of ICP is directly correlated to the accuracy of this transformation
        :param penalties: Penalties are priors from other sensor sources. They can guide the minimization.
        :return: The first element of the tuple is the transformation results and the second is a list of the debugging info at each iteration
        """
        return  self._compute(read, reference, init_tf, penalties)

    def _compute(self,
                read: [np.ndarray, DataPoints],
                reference: [np.ndarray, DataPoints],
                init_tf: Optional[np.ndarray]=None,
                penalties: List[Penalty]=[]) -> Tuple[np.ndarray, List]:

        if isinstance(read, np.ndarray):
            read = DataPoints.from_numpy(read)
        if isinstance(reference, np.ndarray):
            reference = DataPoints.from_numpy(reference)

        dim_read = read.shape[0]
        dim_ref = reference.shape[0]
        if dim_read != dim_ref:
            raise TypeError("The reading and reference DataPoints don't have the same number of rows"
                            f" ({dim_read} != {dim_ref})")
        if init_tf is None:
            init_tf = np.identity(dim_read)

        if not (init_tf.shape[0] == init_tf.shape[1] == dim_read):
            raise TypeError("The 'initial transformation' matrix must be NxN. " 
                            " Where N is the number of rows in the reading and the reference."
                            f" init_tf={init_tf.shape}, read={read.shape}, reference={reference.shape}")

        if isinstance(penalties, Penalty):
            penalties = [penalties]
        penalties_tuples = [p.to_tuple() for p in penalties]
        tf, iterations = self.icp.compute(read.raw_cpp_dp,
                                          reference.raw_cpp_dp,
                                          init_tf,
                                          penalties_tuples,
                                          self._is_dump_enable,
                                          self.dump_config)

        if tf.shape not in [(3, 3), (4, 4)]:
            raise RuntimeError(f"You don't have a 3x3/4x4 transformations matrix, your is {tf.shape}. " 
                               "You probably used points in a row-wise orientation, instead of columns-wise.")
        return tf, self._convert_iterations_dump(iterations)

    def _convert_iterations_dump(self, iterations):
        # I have not yet find a way to convert object from cpp to DataPoints with having to do the conversion twice
        if iterations is None:
            return None
        for iter in iterations:
            for key, value in iter.items():
                if isinstance(value, core.DataPoints):
                    iter[key] = DataPoints(value)
        return iterations