#include "in_memory_inspector.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

using namespace std;

template<typename T>
	InMemoryInspector<T>::InMemoryInspector(const Parameters& params) :
	Inspector("InMemoryInspector",  InMemoryInspector::availableParameters(), params),
	dumpTf(Parametrizable::get<bool>("dumpTf")),
	dumpReference(Parametrizable::get<bool>("dumpReference")),
	dumpReading(Parametrizable::get<bool>("dumpReading")),
	dumpMatches(Parametrizable::get<bool>("dumpMatches")),
	dumpWeights(Parametrizable::get<bool>("dumpWeights"))
	{
	}

template<typename T>
void InMemoryInspector<T>::addStat(const std::string& name, double data)
{
    if (stats.find(name) == stats.end()) {
        stats.insert(std::make_pair(name, vector<double>()));
    }
    stats[name].push_back(data);
}

template<typename T>
void InMemoryInspector<T>::dumpIteration(
				const size_t iterationNumber,
				const TransformationParameters& tfParameters,
				const DataPoints& filteredReference,
				const DataPoints& filteredRead,
				const Matches& matches,
				const OutlierWeights& outlierWeights,
				const TransformationCheckers& transCheck)
{
	Iteration iter = {};
	if (dumpTf)
	    iter.tfParameters = tfParameters;
	if (dumpReference)
	    iter.filteredReference = filteredReference;
	if (dumpReading)
	    iter.filteredRead = filteredRead;
	if (dumpMatches)
	    iter.matches = matches;
	if (dumpWeights)
	    iter.outlierWeights = outlierWeights;

    iterations.push_back(iter);
}


template struct InMemoryInspector<float>;
template struct InMemoryInspector<double>;

