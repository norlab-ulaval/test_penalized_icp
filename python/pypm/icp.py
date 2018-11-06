from typing import Dict

import numpy as np
import yaml

import pypm.pypm_core as core
from pypm.data_points import DataPoints


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
                init_tf: np.ndarray=np.identity(4)) -> np.ndarray:
        if isinstance(read, np.ndarray):
            read = DataPoints.from_numpy(read)
        if isinstance(reference, np.ndarray):
            reference = DataPoints.from_numpy(reference)
        tf = self.icp.compute(read.raw_cpp_dp, reference.raw_cpp_dp, init_tf)

        if tf.shape not in [(3, 3), (4, 4)]:
            raise RuntimeError("You don't have a 3x3/4x4 transformations matrix. " 
                               "You probably used points in a row-wise orientation, instead of colomns-wise.")
        return tf

