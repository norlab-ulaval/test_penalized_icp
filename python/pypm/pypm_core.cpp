
#include <exception>
#include <sstream>
#include <list>

#include <boost/format.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/tuple.hpp>
#include <Eigen/Core>

#include <pointmatcher/PointMatcher.h>

// Remove later on, only used for debug
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include "in_memory_inspector.h"


namespace p = boost::python;
namespace np = boost::python::numpy;

namespace PMS = PointMatcherSupport;
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

		np::ndarray get_features() {
			return eigen_matrix_to_ndarray(this->dp.features);
		}

		np::ndarray get_descriptors() {
			return eigen_matrix_to_ndarray(this->dp.descriptors);
		}

		p::list get_descriptor_labels() {
		    p::list labels;
            for (const PM::DataPoints::Label& l : this->dp.descriptorLabels) {
                labels.append(p::make_tuple(l.text, l.span));
            }

			return labels;
		}

//		np::ndarray get_descriptor_labels() {
//			return eigen_matrix_to_ndarray(this->dp.descriptorLabels);
//		}

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

	p::list from_iterations_stats_to_dict(const std::vector<InMemoryInspector<double>::Iteration> raw_iterations) {
        p::list iterations;
        for (auto& raw_iteration : raw_iterations) {
            p::list iter;

            if (raw_iteration.tfParameters) {
                auto value = eigen_matrix_to_ndarray(raw_iteration.tfParameters.value());
                iter.append(p::make_tuple("tf", value));
            }
            if (raw_iteration.filteredReference) {
                auto value = DataPoints(raw_iteration.filteredReference.value());
                iter.append(p::make_tuple("filtered_ref", value));
            }
            if (raw_iteration.filteredRead) {
                auto value = DataPoints(raw_iteration.filteredRead.value());
                iter.append(p::make_tuple("filtered_read", value));
            }
//            if (raw_iteration.matches) {
//                auto value = eigen_matrix_to_ndarray(raw_iteration.matches.value());
//                iter.append(p::make_tuple("matches", value));
//            }
            if (raw_iteration.outlierWeights) {
                auto value = eigen_matrix_to_ndarray(raw_iteration.outlierWeights.value());
                iter.append(p::make_tuple("outlier_weight", value));
            }

            iterations.append(p::dict(iter));
        }
        return iterations;
    }

    std::shared_ptr<InMemoryInspector<double>> inspector_factory(const p::dict dump_config) {
        PMS::Parametrizable::Parameters params;
        const std::map<std::string, std::string> params_name_mapping = {
            {"tf", "dumpTf"},
            {"matches", "dumpMatches"},
            {"outlier_weight", "dumpWeights"},
            {"filtered_ref", "dumpReference"},
            {"filtered_read", "dumpReading"}
        };
        for (const auto& param_name: params_name_mapping) {
            const std::string value = p::extract<bool>(dump_config.get(param_name.first)) ? "1" : "0";
            params.insert(std::make_pair(param_name.second, value));
        }
        //std::cout << "Exist" << params_name_mapping << std::endl;
        std::shared_ptr<InMemoryInspector<double>> inspector(new InMemoryInspector<double>(params));

        return inspector;
    }

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

            np::ndarray compute_residual_function(const DataPoints& reference_in,
                                                  const p::list mins,
                                                  const p::list maxs,
                                                  const p::list nb_samples) {

	            PM::DataPoints reference(reference_in.get_dp());

                if (false) {
                    Eigen::MatrixXd point = Eigen::MatrixXd::Ones(3, 1);
                    point(0, 0) = 0;
                    point(1, 0) = 0;
                    reference = eigen_to_datapoints(point);
                    Eigen::MatrixXd oriC(2, 2);
                    oriC(0, 0) = 0.001; oriC(0, 1) = 0;
                    oriC(1, 0) = 0; oriC(1, 1) = 5;

                    double th = 55 * M_PI / 180.0;
                    Eigen::Matrix2d R(2, 2);
                    R << cos(th), -sin(th),
                         sin(th), +cos(th);


                    Eigen::MatrixXd C = R * oriC * R.transpose();

                    const Eigen::EigenSolver<PM::Matrix> solver(C);
                    const PM::Vector eigenVal = solver.eigenvalues().real();
                    const PM::Matrix eigenVec = solver.eigenvectors().real();
		            const PM::Vector eigenVec_inVec = Eigen::Map<const PM::Vector>(eigenVec.data(), 2 * 2);

	                reference.allocateDescriptor("eigVectors", 2 * 2);
	                reference.allocateDescriptor("eigValues", 1 * 2);
	                reference.getDescriptorViewByName("eigValues").col(0) = eigenVal;
	                reference.getDescriptorViewByName("eigVectors").col(0) = eigenVec_inVec;
                    std::cout << "oriC:" << std::endl << oriC << std::endl;
                    std::cout << "C :" << std::endl << C  << std::endl;
                    std::cout << "R :" << std::endl << R  << std::endl;
                    std::cout << "eigenVal:" << std::endl << eigenVal << std::endl;
                    std::cout << "eigenVec:" << std::endl << eigenVec << std::endl;
                    std::cout << "eigenVec_inVec:" << std::endl << eigenVec_inVec << std::endl;

                } else {
                    this->icp.referenceDataPointsFilters.init();
                    this->icp.referenceDataPointsFilters.apply(reference);
                }

                // Delete all points except the first one
                if (false) {
                    reference.features = reference.features.col(13);
                    reference.descriptors = reference.descriptors.col(13);
                }

                this->icp.matcher->init(reference);

                PM::DataPoints read_in = eigen_to_datapoints(Eigen::MatrixXd(3, 1));
                const PM::OutlierWeights weight = Eigen::MatrixXd::Ones(1, 1);
                // TODO: add support for more than 2 dimensions
                int nb_samples_x = p::extract<int>(nb_samples[0]);
                int nb_samples_y = p::extract<int>(nb_samples[1]);
                const Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(nb_samples_x,
                                                                      p::extract<float>(mins[0]),
                                                                      p::extract<float>(maxs[0]));
                const Eigen::VectorXd ys = Eigen::VectorXd::LinSpaced(nb_samples_y,
                                                                      p::extract<float>(mins[1]),
                                                                      p::extract<float>(maxs[1]));
                p::list residuals;
                for (size_t j = 0; j < nb_samples_y; ++j) {
                    for (size_t i = 0; i < nb_samples_x; ++i) {
                        read_in.features(0, 0) = xs(i);
                        read_in.features(1, 0) = ys(j);
                        read_in.features(2, 0) = 1;
                        const PM::Matches matches(this->icp.matcher->findClosests(read_in));
                        const double residual = this->icp.errorMinimizer->getResidualError(read_in, reference, weight, matches, {},  PM::Matrix());
                        residuals.append(p::make_tuple(xs(i), ys(j), residual));
                    }
                }
                return np::array(residuals);
            }
            p::tuple compute(const DataPoints& read_in,
                                const DataPoints& reference_in,
                                const np::ndarray& init_tf_nd,
                                const p::list& penalty_nd,
                                const bool dump_enable,
                                const p::dict dump_config) {
                auto init_tf = ndarray_to_eigen_matrix<double>(init_tf_nd);
                auto penalties = convert_penalties_to_eigen(penalty_nd);
                if (dump_enable) {
                    if (this->icp.inspector && this->icp.inspector->className != "NullInspector") {
                        throw std::runtime_error("An inspector has already been configured, can not add the InMemoryInspector. "
                        "To dump info during the registration a InMemoryInspector is used, it overrides the current inspector. "
                        "Please change your inspector to a NullInspector.");
                    }
                    std::shared_ptr<InMemoryInspector<double>> inspector = inspector_factory(dump_config);
                    auto old_inspector = this->icp.inspector;
                    this->icp.inspector = inspector;

                    const auto tf = eigen_matrix_to_ndarray(this->icp.compute(read_in.get_dp(),
                                                                              reference_in.get_dp(),
                                                                              init_tf,
                                                                              penalties));

                    auto dump_info = from_iterations_stats_to_dict(inspector->iterations);
                    this->icp.inspector.swap(old_inspector);
                    return p::make_tuple(tf, dump_info);
                } else {
                    const auto tf = eigen_matrix_to_ndarray(this->icp.compute(read_in.get_dp(),
                                                                              reference_in.get_dp(),
                                                                              init_tf,
                                                                              penalties));

                    return p::make_tuple(tf, p::object());  // p::object() == None for python
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
    .def("compute_residual_function", &Wrapper::ICP::compute_residual_function)
    ;

  p::def("from_ndarray_to_datapoint", &Wrapper::from_ndarray_to_datapoint);

	p::class_<Wrapper::DataPoints>("DataPoints")
					.def("get_features", &Wrapper::DataPoints::get_features)
					.def("get_descriptors", &Wrapper::DataPoints::get_descriptors)
					.def("get_descriptor_labels", &Wrapper::DataPoints::get_descriptor_labels)
					//.def("get_descriptor_labels", &Wrapper::DataPoints::get_descriptor_labels)
					.def("get_shape", &Wrapper::DataPoints::get_shape)
					;
}
