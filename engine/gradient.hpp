#ifndef GRADIENT_H_
#define GRADIENT_H_

// Engine headers
#include <matrix.hpp>
#include <erf.hpp>
#include <vector.hpp>
#include <layer.hpp>

namespace zhetapi {

namespace ml {

template <class T>
Vector <T> simple_compute_cached(
		Layer <T> *layers,
		size_t size,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in)
{
	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < size) {
		a[i] = tmp.append_above(T (1));

		layers[i].forward_propogate(tmp, prv);

		z[i] = layers[i].__dact->compute(prv);

		i++;
	}

	a[i] = tmp;

	return tmp;
}

template <class T>
Matrix <T> *simple_gradient(
		Layer <T> *layers,
		size_t size,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in,
		const Vector <T> &out,
		Erf <T> *opt)
{
	// Compute the actual value
	Vector <T> actual = simple_compute_cached(layers, size, a, z, in);

	// Get the derivative of the cost
	Erf <T> *dopt = opt->derivative();

	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [size];

	Vector <T> delta = dopt->compute(out, actual);
	for (int i = size - 1; i >= 0; i--) {
		if (i < size - 1)
			delta = std::move(rmt_and_mult(layers[i + 1].__mat, delta));

		delta.stable_shur(z[i]);

		J[i] = std::move(vvt_mult(delta, a[i]));
	}

	// Free resources
	delete dopt;

	// Return the gradient
	return J;
}

}

}

#endif
