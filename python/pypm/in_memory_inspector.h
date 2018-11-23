#ifndef __POINTMATCHER_IN_MEMORY_INSPECTORS_H
#define __POINTMATCHER_IN_MEMORY_INSPECTORS_H

#include <pointmatcher/PointMatcher.h>

/**
 * Update 2018-11-08: The module have not been added to libpointmatcher for the time being.
 *
 * Why is InMemoryInspector not in registry and not in InspectorImpl?
 * Contrary to other module of libpointmatcher, InMemoryInspector is a module that should only
 * be config into Libpm programmatically.  The goal of this inspector is to log all the stats
 * and copy them into memory. Then the user can read those stats directly into memory, this
 * remove the requirement of dump the stats to the disk and then parsing the dump.
 *
 * The user must have access to the implementation of InMemoryInspector otherwise it can not
 * read it's member function and get the stats. In libpointmatcher the implementation of other
 * modules are not public to the user.
 */
template<typename T>
struct InMemoryInspector: public PointMatcher<T>::Inspector
{
public:
	 typedef PointMatcherSupport::Parametrizable Parametrizable;
	 typedef Parametrizable::Parameters Parameters;
	 typedef Parametrizable::ParameterDoc ParameterDoc;
	 typedef Parametrizable::ParametersDoc ParametersDoc;

	 typedef typename PointMatcher<T>::Inspector Inspector;
	 typedef typename PointMatcher<T>::DataPoints DataPoints;
	 typedef typename PointMatcher<T>::Matches Matches;
	 typedef typename PointMatcher<T>::OutlierWeights OutlierWeights;
	 typedef typename PointMatcher<T>::TransformationParameters TransformationParameters;
	 typedef typename PointMatcher<T>::TransformationCheckers TransformationCheckers;
	 typedef typename PointMatcher<T>::Matrix Matrix;

	inline static const std::string description()
	{
		return "Keep in memory statistics at each step.";
	}

	struct IterationData {
		TransformationParameters tfParameters;
	};

	std::vector<IterationData> iterationsStats;
	std::map<std::string, std::vector<double>> stats;

	InMemoryInspector() : Inspector("InMemoryInspector",  ParametersDoc(), Parameters()) {}
    // TODO: Add a way to enable/disable the dumping of each of those field
	virtual void dumpIteration(const size_t iterationNumber,
                               const TransformationParameters& parameters,
                               const DataPoints& filteredReference,
                               const DataPoints& reading,
                               const Matches& matches,
                               const OutlierWeights& outlierWeights,
                               const TransformationCheckers& transformationCheckers);
	virtual void addStat(const std::string& name, double data);

};


#endif // __POINTMATCHER_IN_MEMORY_INSPECTORS_H
