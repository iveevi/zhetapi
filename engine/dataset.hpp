#ifndef DATASET_H_
#define DATASET_H_

#ifndef __AVR

// C/C++ headers
#include <vector>

// Engine headers
#include "vector.hpp"

namespace zhetapi {

template <class T>
using DataSet = std::vector <Vector <T>>;

template <class T>
std::vector <DataSet <T>> split(const DataSet <T> &dset, size_t len)
{
	std::vector <DataSet <T>> batched;

	DataSet <T> batch;

	size_t size = dset.size();
	for (int i = 0; i < size; i++) {
		batch.push_back(dset[i]);

		if (i % len == len - 1 || i == size - 1) {
			batched.push_back(batch);

			batch.clear();
		}
	}

	return batched;
}

// General sets of N-dimensional data
template <class T, size_t N>
class NumericalData {
	// TODO: make more efficient?
	Vector <T> _stddev() {
		Vector <T> sum(N, 0);
		for (const auto &vec : dataset) {
			Vector <T> dx = (vec - _mean);
			sum += shur(dx, dx);
		}
		return sum/(dataset.size() - _sample);
	}

	// TODO: fixed vector?
	Vector <T> _mean() {
		Vector <T> sum(N, 0);
		for (const auto &vec : dataset)
			sum += vec;
		return sum/dataset.size();
	}
public:
	DataSet		dataset;
	Vector <T>	mean;
	Vector <T>	stddev;
	bool		sample;

	// Sample indicates whether the dataset is a
	// sample or the entire population
	Data(const DataSet &set, bool sample = false)
		: dataset(set), mean(_mean()),
		stddev(_stddev()), sample(sample) {}
};

// Dimensional
template <class T>
using BivariateData <T> = NumericalData <T, 2>;

}

#else

#warning Zhetapi does not support zhetapi::Dataset for AVR systems.

#endif

#endif
