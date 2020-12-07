#ifndef DATASET_H_
#define DATASET_H_

// C/C++ headers
#include <vector>

// Engine headers
#include <vector.hpp>

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
	
}

#endif
