
#include <exception>
#include <sstream>

#include <boost/format.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/tuple.hpp>
#include <Eigen/Core>

#include <pointmatcher/PointMatcher.h>


namespace p = boost::python;
namespace np = boost::python::numpy;

typedef PointMatcher<double> PM;

// Fix size
template <typename T, int I, int J>
Eigen::Matrix<T,I,J> ndarray_to_eigen_matrix(const np::ndarray& np_m) {
  Eigen::Matrix<T,I,J> eigen_m;

  for(auto i = 0; i < I; i++) {
    for(auto j = 0; j < J; j++) {
      eigen_m(i,j) = p::extract<T>(np_m[i][j]);
    }
  }

  return eigen_m;
}

// Dynamic size
template <typename T>
Eigen::MatrixXd ndarray_to_eigen_matrix(const np::ndarray& np_m) {
	const int I = p::extract<int>(np_m.attr("shape")[0]);
	const int J = p::extract<int>(np_m.attr("shape")[1]);
	Eigen::MatrixXd eigen_m(I, J);

	for(auto i = 0; i < I; i++) {
		for(auto j = 0; j < J; j++) {
			eigen_m(i,j) = p::extract<T>(np_m[i][j]);
		}
	}

	return eigen_m;
}

// Fix size
template <typename T, int I, int J>
np::ndarray eigen_matrix_to_ndarray(const Eigen::Matrix<T,I,J>& eigen_m) {
  p::list matrix;

  for(auto i = 0; i < I; i++) {
    p::list row;
    for(auto j = 0; j < J; j++) {
      row.append(eigen_m(i,j));
    }
    matrix.append(row);
  }

  return np::array(matrix);
}

// Dynamic size
np::ndarray eigen_matrix_to_ndarray(const Eigen::MatrixXd& eigen_m) {
	p::list matrix;

	for(auto i = 0; i < eigen_m.rows(); i++) {
		p::list row;
		for(auto j = 0; j < eigen_m.cols(); j++) {
			row.append(eigen_m(i,j));
		}
		matrix.append(row);
	}

	return np::array(matrix);
}


PM::DataPoints eigen_to_datapoints(const Eigen::MatrixXd& mat) {

	PM::DataPoints::Labels features;
	features.push_back(PM::DataPoints::Label("x", 1));
	features.push_back(PM::DataPoints::Label("y", 1));
	features.push_back(PM::DataPoints::Label("z", 1));
	features.push_back(PM::DataPoints::Label("pad", 1));

	return PM::DataPoints(mat, features);
}

namespace Wrapper {

	class DataPoints {
	public:
		DataPoints(): dp() {}
		DataPoints(const PM::DataPoints& p_dp): dp(p_dp) {}
		np::ndarray to_numpy() {
			std::cout << this->dp.features << std::endl;
			return eigen_matrix_to_ndarray(this->dp.features);
		}

		const PM::DataPoints& get_dp() const {
			return this->dp;
		}

	private:
		PM::DataPoints dp;
	};

	DataPoints from_ndarray_to_datapoint(const np::ndarray& mat) {
		return DataPoints(eigen_to_datapoints(ndarray_to_eigen_matrix<double>(mat)));
	}

	class ICP {
		private:
			PM::ICP icp;
			public:
				ICP(): icp() {}

				void set_default() {
					this->icp.setDefault();
				}

				void load_from_yaml(std::string yaml_content) {
					std::istringstream yaml_content_stream(yaml_content);
					this->icp.loadFromYaml(yaml_content_stream);
				}

				np::ndarray compute(const DataPoints& read_in,
														const DataPoints& reference_in,
														const np::ndarray& init_tf_nd) {
					auto init_tf = ndarray_to_eigen_matrix<double>(init_tf_nd);
					const Eigen::MatrixXd tf = this->icp.compute(read_in.get_dp(), reference_in.get_dp(), init_tf);
					return eigen_matrix_to_ndarray(tf);
			}
	};

}

BOOST_PYTHON_MODULE(pypm_core) {
  np::initialize();
  p::class_<Wrapper::ICP>("ICP")
    .def("set_default", &Wrapper::ICP::set_default)
    .def("load_from_yaml", &Wrapper::ICP::load_from_yaml)
    .def("compute", &Wrapper::ICP::compute)
    ;

  p::def("from_ndarray_to_datapoint", &Wrapper::from_ndarray_to_datapoint);

	p::class_<Wrapper::DataPoints>("DataPoints")
					.def("to_numpy", &Wrapper::DataPoints::to_numpy)
					;
}
