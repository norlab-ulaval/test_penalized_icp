from copy import deepcopy


class ConfigBuilder:
    def __init__(self):
        self._base_config = {}

        self._matcher = {'KDTreeMatcher': {'knn': 1}}
        self._inspector = 'NullInspector'
        self._reading_dp_filter = []
        self._reference_dp_filter = []
        self._outlier_filters = []

        self.with_point_to_point().with_tf_checker()

    def copy(self):
        return deepcopy(self)

    def with_tf_checker(self, knn=40):
        self._tf_checker = [
            {'CounterTransformationChecker': {'maxIterationCount': knn}},
            'DifferentialTransformationChecker'
        ]
        return self

    # Inspector
    def with_vtk_inspector(self):
        self._inspector = {
            "VTKFileInspector": {
                 "baseFileName" : "vissteps",
                 "dumpDataLinks" : 1,
                 "dumpReading" : 1,
                 "dumpReference" : 1
            }
        }
        return self

    # Error minimizer
    def with_point_to_point(self, confidence_in_penalties=0.0):
        self._minimizer = {"PointToPointWithPenaltiesErrorMinimizer": {"confidenceInPenalties": confidence_in_penalties}}
        return self

    def with_point_to_plane(self):
        self._minimizer = {"PointToPlaneWithPenaltiesErrorMinimizer": {}}
        return self

    def with_point_to_gaussian(self):
        self._minimizer = {"PointToGaussianErrorMinimizer": {}}
        return self

    # Datafilters
    def add_normal_to_read(self, knn=5):
        self._reading_dp_filter.append({"SurfaceNormalDataPointsFilter": {"knn": knn}})
        return self

    def add_normal_to_ref(self, knn=5):
        self._reference_dp_filter.append({"SurfaceNormalDataPointsFilter": {"knn": knn}})
        return self

    # def add_cov_to_read(self, knn=5):
    #     self._reading_dp_filter.append({"SurfaceCovarianceDataPointsFilter": {"knn": knn}})
    #     return self

    def add_cov_to_ref(self, knn=5):
        self._reference_dp_filter.append({"SurfaceCovarianceDataPointsFilter": {"knn": knn}})
        return self

    # def add_decompose_cov_to_read(self, keep_normals=False):
    #     normal = 1 if keep_normals else 0
    #     self._reading_dp_filter.append({"DecomposeCovarianceDataPointsFilter": {"keepNormals": normal}})
    #     return self

    def add_decompose_cov_to_ref(self, keep_normals=False):
        normal = 1 if keep_normals else 0
        self._reference_dp_filter.append({"DecomposeCovarianceDataPointsFilter": {"keepNormals": normal}})
        return self

    def add_sensor_noise_to_read(self, sensor_type=0, generate_cov=False):
        cov = 1 if generate_cov else 0
        self._reading_dp_filter.append({"SimpleSensorNoiseDataPointsFilter": {"sensorType": sensor_type,
                                                                              "covariance": cov}})
        return self

    def add_sensor_noise_to_ref(self, sensor_type=0, generate_cov=False):
        cov = 1 if generate_cov else 0
        self._reference_dp_filter.append({"SimpleSensorNoiseDataPointsFilter": {"sensorType": sensor_type,
                                                                                "covariance": cov}})
        return self

    # Outlier filters
    def add_outlier_filter_trim(self, overlap=0.75):
        self._outlier_filters.append({"TrimmedDistOutlierFilter": {"ratio": overlap}})
        return self

    def add_outlier_filter_sensor_noise(self):
        self._outlier_filters.append({"SensorNoiseOutlierFilter": {}})
        return self

    def build(self):
        return  {
            'outlierFilters': self._outlier_filters,
            'errorMinimizer': self._minimizer,
            'matcher': self._matcher,
            'transformationCheckers': self._tf_checker,
            'readingDataPointsFilters': self._reading_dp_filter,
            'referenceDataPointsFilters': self._reference_dp_filter,
            'inspector': self._inspector
        }