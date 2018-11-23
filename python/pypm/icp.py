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

    def compute(self,
                read: [np.ndarray, DataPoints],
                reference: [np.ndarray, DataPoints],
                init_tf: Optional[np.ndarray]=None,
                penalties: List[Penalty]=[]) -> np.ndarray:
        tf, _ =  self._compute(read, reference, init_tf, penalties, dump_info=False)
        return tf

    def compute_with_dump_info(self,
                read: [np.ndarray, DataPoints],
                reference: [np.ndarray, DataPoints],
                init_tf: Optional[np.ndarray]=None,
                penalties: List[Penalty]=[]) -> Tuple[np.ndarray, List]:
        return self._compute(read, reference, init_tf, penalties, dump_info=True)

    def _compute(self,
                read: [np.ndarray, DataPoints],
                reference: [np.ndarray, DataPoints],
                init_tf: Optional[np.ndarray]=None,
                penalties: List[Penalty]=[],
                dump_info: bool=False) -> Tuple[np.ndarray, List]:

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
        tf, iter_stats = self.icp.compute(read.raw_cpp_dp, reference.raw_cpp_dp, init_tf, penalties_tuples, dump_info)

        if tf.shape not in [(3, 3), (4, 4)]:
            raise RuntimeError(f"You don't have a 3x3/4x4 transformations matrix, your is {tf.shape}. " 
                               "You probably used points in a row-wise orientation, instead of columns-wise.")
        return tf, iter_stats

