#ifndef FILTER_H_
#define FILTER_H_

// Engine headers
#include <tensor.hpp>

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
	virtual void forward_propogate(const Pipe <T> &, Pipe <T> &) = 0;

	// Is this the right type?
	// virtual const std::vector <Tensor <T> *> &back_propogate(const Pipe &) const = 0;
};

}

}

#endif
