
#include <exception>
#include <sstream>
#include <list>

#include <boost/format.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/tuple.hpp>
#include <Eigen/Core>

#include <pointmatcher/PointMatcher.h>

#include "in_memory_inspector.h"


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
	if (mat.rows() == 4) { // TODO make this cleaner, a loop over x, y, z
	    features.push_back(PM::DataPoints::Label("z", 1));
	}
	features.push_back(PM::DataPoints::Label("pad", 1));

	return PM::DataPoints(mat, features);
}

namespace Wrapper {


	class DataPoints {
	public:
		DataPoints(): dp() {}
		DataPoints(const PM::DataPoints& p_dp): dp(p_dp) {}
		np::ndarray to_numpy() {
			return eigen_matrix_to_ndarray(this->dp.features);
		}

		const p::tuple get_shape() const {
		    return p::make_tuple(this->dp.features.rows(), this->dp.features.cols());
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

	p::list from_iterations_stats_to_dict(const std::vector<InMemoryInspector<double>::IterationData> raw_iter_stats,
                                          const Eigen::MatrixXd init_tf) {
        p::list iter_stats;
        for (auto& raw_iter_stat : raw_iter_stats) {
            p::list iter_stat;

            auto tf = eigen_matrix_to_ndarray(raw_iter_stat.tfParameters);
            iter_stat.append(p::make_tuple("tf", tf));

            iter_stats.append(p::dict(iter_stat));
        }
        return iter_stats;
    }
//std::vector<std::tuple<np::ndarray, np::ndarray>>&
    PM::ErrorMinimizer::Penalties convert_penalties_to_eigen(const p::list& penalty_nd) {
        PM::ErrorMinimizer::Penalties penalties;

        // Convert python list to std list
        p::stl_input_iterator<p::tuple> begin(penalty_nd), end;
        auto penalties_list = std::list<p::tuple>(begin, end);

        penalties.reserve(penalties_list.size());
        for (auto& p : penalties_list) {
            penalties.push_back(std::make_pair(ndarray_to_eigen_matrix<double>(p::extract<np::ndarray>(p[0])),
                                               ndarray_to_eigen_matrix<double>(p::extract<np::ndarray>(p[1]))));
        }
        return penalties;
    }

	class ICP {
        public:
            ICP(): icp() {}

            void set_default() {
                this->icp.setDefault();
            }

            void load_from_yaml(std::string yaml_content) {
                std::istringstream yaml_content_stream(yaml_content);
                this->icp.loadFromYaml(yaml_content_stream);
            }


            p::tuple compute(const DataPoints& read_in,
                                const DataPoints& reference_in,
                                const np::ndarray& init_tf_nd,
                                const p::list& penalty_nd,
                                const bool dump_info) {
                auto init_tf = ndarray_to_eigen_matrix<double>(init_tf_nd);
                auto penalties = convert_penalties_to_eigen(penalty_nd);
                if (dump_info) {
                    if (this->icp.inspector && this->icp.inspector->className != "NullInspector") {
                        throw std::runtime_error("An inspector has already been configured, can not add the InMemoryInspector. "
                        "To dump info during the registration a InMemoryInspector is used, it overides the current inspector. "
                        "Please change your inspector to a NullInspector.");
                    }
                    std::shared_ptr<InMemoryInspector<double>> inspector(new InMemoryInspector<double>());
                    auto old_inspector = this->icp.inspector;
                    this->icp.inspector = inspector;

                    const auto tf = eigen_matrix_to_ndarray(this->icp.compute(read_in.get_dp(),
                                                                              reference_in.get_dp(),
                                                                              init_tf,
                                                                              penalties));

                    auto dump_info = from_iterations_stats_to_dict(inspector->iterationsStats, init_tf);
                    this->icp.inspector.swap(old_inspector);
                    return p::make_tuple(tf, dump_info);
                } else {

                    const auto tf = eigen_matrix_to_ndarray(this->icp.compute(read_in.get_dp(),
                                                                              reference_in.get_dp(),
                                                                              init_tf,
                                                                              penalties));

                    return p::make_tuple(tf, p::object());
                }
            }
		private:

			PM::ICP icp;
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
					.def("get_shape", &Wrapper::DataPoints::get_shape)
					;
}
