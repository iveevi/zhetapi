#ifndef STD_OPTIMIZERS_H_
#define STD_OPTIMIZERS_H_

// Engine headers
#include <optimizer.hpp>

namespace zhetapi {

namespace ml {

template <class T>
class DefaultOptimizer : public Optimizer <T> {
	T	__alpha;
public:
	DefaultOptimizer(T alpha) : __alpha(alpha) {}

	Matrix <T> *gradient(Layer <T> *layers, size_t size, const Vector <T> &in, const Vector <T> &out, Erf <T> *cost) {
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];

    		Matrix <T> *J = simple_gradient(layers, size, a, z, in, out, cost);
		
		delete[] a;
		delete[] z;

		return J;
	}
};

}

}

#endif