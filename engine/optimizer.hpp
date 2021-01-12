#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// Engine headers
#include <gradient.hpp>

namespace zhetapi {

namespace ml {

// Optimizer class
template <class T>
class Optimizer {
public:
	virtual Matrix <T> *gradient(Layer <T> *, size_t, const Vector <T> &, const Vector <T> &, Erf <T> *) = 0;
};

}

}

#endif
