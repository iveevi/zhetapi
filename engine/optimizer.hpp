#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// Engine headers
#include <gradient.hpp>
#include <dataset.hpp>

namespace zhetapi {

namespace ml {

// Optimizer class
template <class T>
class Optimizer {
public:
	virtual ~Optimizer();

	virtual Matrix <T> *gradient(
			Layer <T> *,
			size_t,
			const Vector <T> &,
			const Vector <T> &,
			Erf <T> *) = 0;

	// TODO: Make pure virtual
	virtual Matrix <T> *gradient(Layer <T> *, size_t, const DataSet <T> &, const DataSet <T> &, Erf <T> *) {return nullptr;};
};

template <class T>
Optimizer <T> ::~Optimizer() {}

}

}

#endif
