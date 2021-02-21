#ifndef FILTER_H_
#define FILTER_H_

// Engine headers
#include <tensor.hpp>

namespace zhetapi {

template <class T = double>
class Filter {
public:
	/*
	 * Process method:
	 *
	 * Takes in a set of inputs, performs the necessary computations, and
	 * places the results into the locations specified by the second vector
	 * of pointers. Notes that the inputs are also passed as a list of
	 * pointers.
	 */
	virtual process(const std::vector <Tensor <T> *> &,
			const std::vector <Tensor <T> *> &) const = 0;
};

}

#endif
