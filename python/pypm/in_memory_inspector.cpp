#include "in_memory_inspector.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

using namespace std;

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
				const DataPoints& reading,
				const Matches& matches,
				const OutlierWeights& outlierWeights,
				const TransformationCheckers& transCheck)
{
	IterationData data = {
		tfParameters
	};
    iterationsStats.push_back(data);
}


template struct InMemoryInspector<float>;
template struct InMemoryInspector<double>;

