#ifndef FILTER_H_
#define FILTER_H_

// Engine headers
#include "tensor.hpp"

namespace zhetapi {

namespace ml {

// Type aliases
template <class T>
using Pipe = std::vector <Tensor <T> *>;

template <class T = double>
class Filter {
public:
	/**
	 * @brief Process method: takes in a set of inputs, performs the
	 * necessary computations, and places the results into the locations
	 * specified by the second vector of pointers. Notes that the inputs
	 * are also passed as a list of pointers.
	 */
	virtual void propogate(const Pipe <T> &, Pipe <T> &) = 0;
	virtual void gradient(const Pipe <T> &, Pipe <T> &) = 0;
	virtual void apply_gradient(const Pipe <T> &) = 0;
};

}

}

#endif
