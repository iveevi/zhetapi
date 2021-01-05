#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// Engine headers
#include <matrix.hpp>

namespace zhetapi {

namespace ml {

// Basic computation
template <class T>
Vector <T> compute(
		const Vector <T> &in,
		Matrix <T> *weights,
		Activation <T> *acts,
		size_t size,
		Vector <T> *a,
		Vector <T> *z)
{

	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < size - 1) {
		a[i] = tmp.append_above(T (1));

		prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

		tmp = acts[i + 1]->compute(prv);

		Activation <T> *act = acts[i + 1]->derivative();

		z[i++] = (*act)(prv);

		delete act;
	}

	a[i] = tmp;
	
	return tmp;
}

// Basic backpropogation with caching
template <class T>
Matrix <T> *simple_gradient(
		Matrix <T> *weights,
		Activation <T> *acts,
		size_t size,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in,
		const Vector <T> &out)
{
	// Compute the actual value
	Vector <T> actual = compute(in, weights, acts, size, a, z);
	
	// Get the derivative of the cost
	Erf <T> *dopt = opt->derivative();
	
	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [__size - 1];

	Vector <T> delta = dopt->compute(out, actual);
	for (int i = __size - 2; i >= 0; i--) {
		if (i < __size - 2)
			delta = std::move(rmt_and_mult(weights[i + 1], delta));
		
		delta.stable_shur(z[i]);

		J[i] = std::move(vvt_mult(delta, a[i]));
	}

	// Free resources
	delete dopt;

	// Return the gradient
	return J;
}

// Optimizer class
template <class T>
class Optimizer {
public:
	virtual Matrix <T> *gradient(Matrix <T> *) = 0;
};

}

}

#endif
