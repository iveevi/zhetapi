#ifndef STD_OPTIMIZERS_H_
#define STD_OPTIMIZERS_H_

// Engine headers
#include <optimizer.hpp>

namespace zhetapi {

namespace ml {

/*
 * This header contains the standard optimizers.
 *
 * Preferably use one optimizer per neural network.
 */

template <class T>
class SGD : public Optimizer <T> {
	T	__alpha;
public:
	SGD(T alpha = 0.001) : __alpha(alpha) {}

	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];

    		Matrix <T> *J = simple_gradient(layers, size, a, z, in, out, cost);

		for (size_t i = 0; i < size; i++)
			J[i] *= __alpha;
		
		delete[] a;
		delete[] z;

		return J;
	}
};

/*
 * TODO: Cache J in the optimizer itself so that we do not need to repeatedly
 * allocate and free the gradient.
 */
template <class T>
class Momentum : public Optimizer <T> {
	T		__alpha = 0;
	T		__beta = 0;

	Matrix <T> *	__M = nullptr;
	size_t		__size = 0;
public:
	Momentum(T alpha = 0.001, T beta = 0.9) : __alpha(alpha), __beta(beta) {}

	~Momentum() {
		delete[] __M;
	}

	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];

    		Matrix <T> *J = simple_gradient(layers, size, a, z, in, out, cost);
		
		if (size != __size) {
			delete[] __M;

			__size = size;
			__M = new Matrix <T> [__size];
		}
	
		Matrix <T> *Jo = new Matrix <T> [size];
		for (size_t i = 0; i < __size; i++) {
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (__M[i].get_dimensions() != odim)
				__M[i] = Matrix <T> (odim.first, odim.second, T(0));
			
			Jo[i] = __M[i] = __beta * __M[i] + __alpha * J[i];
		}
		
		delete[] a;
		delete[] z;
		delete[] J;

		return Jo;
	}

	void reset() {
		delete[] __M;

		__size = 0;
		__M = nullptr;
	}
};

template <class T>
class Nesterov : public Optimizer <T> {
	T		__alpha;
	T		__beta;

	Matrix <T> *	__M = nullptr;
	size_t		__size = 0;
public:
	Nesterov(T alpha = 0.001, T beta = 0.9) : __alpha(alpha), __beta(beta) {}

	~Nesterov() {
		delete[] __M;
	}

	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];

		Layer <T> *adj = new Layer <T> [size];
		
		// Ensure that __M is of the right size
		if (size != __size) {
			delete[] __M;

			__size = size;
			__M = new Matrix <T> [__size];
		}

		for (size_t i = 0; i < size; i++) {
			size_t rs = layers[i].get_fan_out();
			size_t cs = layers[i].get_fan_in() + 1;

			if (__M[i].get_dimensions() != std::make_pair(rs, cs))
				__M[i] = Matrix <T> (rs, cs, T(0));

			adj[i] = layers[i] - __M[i];
		}

    		Matrix <T> *J = simple_gradient(adj, size, a, z, in, out, cost);

		Matrix <T> *Jo = new Matrix <T> [size];
		for (size_t i = 0; i < __size; i++)
			Jo[i] = __M[i] = __beta * __M[i] + __alpha * J[i];
		
		delete[] a;
		delete[] z;
		delete[] J;
		delete[] adj;

		return Jo;
	}

	void reset() {
		delete[] __M;

		__size = 0;
		__M = nullptr;
	}
};

template <class T>
class AdaGrad : public Optimizer <T> {
	T		__eta;

	Matrix <T> *	__S = nullptr;
	size_t		__size = 0;
public:
	AdaGrad(T eta = 0.001) : __eta(eta) {}

	~AdaGrad() {
		delete[] __S;
	}

	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];
    		
		Matrix <T> *J = simple_gradient(layers, size, a, z, in, out, cost);

		if (size != __size) {
			delete[] __S;

			__size = size;
			__S = new Matrix <T> [__size];
		}

		using namespace std;
		for (size_t i = 0; i < __size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (__S[i].get_dimensions() != odim) {
				/*cout << "Reseting __S:" << endl;
				cout << "\t" << __S[i] << endl;*/
				__S[i] = Matrix <T> (odim.first, odim.second, T(0));
				// cout << "\t" << __S[i] << endl;
			}

			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			// cout << "S[i] = " << __S[i] << endl;
			__S[i] = __S[i] + shur(J[i], J[i]);

			/* cout << "S'[i] = " << __S[i] << endl;
			cout << "J[i] = " << J[i] << endl;
			cout << "===================================" << endl; */

			J[i] = inv_shur(__eta * J[i], __S[i] + e);
		}

		delete[] a;
		delete[] z;

		return J;
	}
	
	void reset() {
		delete[] __S;

		__size = 0;
		__S = nullptr;
	}

	static const T epsilon;
};

template <class T>
const T AdaGrad <T> ::epsilon = 1e-10;

template <class T>
class RMSProp : public Optimizer <T> {
	T		__alpha;
	T		__beta;

	Matrix <T> *	__S = nullptr;
	size_t		__size = 0;
public:
	RMSProp(T alpha = 0.001, T beta = 0.9) : __alpha(alpha), __beta(beta) {}

	~RMSProp() {
		delete[] __S;
	}

	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];
    		
		Matrix <T> *J = simple_gradient(layers, size, a, z, in, out, cost);

		if (size != __size) {
			delete[] __S;

			__size = size;
			__S = new Matrix <T> [__size];
		}

		for (size_t i = 0; i < __size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (__S[i].get_dimensions() != odim)
				__S[i] = Matrix <T> (odim.first, odim.second, T(0));

			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			__S[i] = __beta * __S[i] + (1 - __beta) * shur(J[i], J[i]);

			J[i] = inv_shur(__alpha * J[i], __S[i] + e);
		}

		delete[] a;
		delete[] z;

		return J;
	}
	
	void reset() {
		delete[] __S;

		__size = 0;
		__S = nullptr;
	}

	static const T epsilon;
};

template <class T>
const T RMSProp <T> ::epsilon = 1e-10;

template <class T>
class Adam : public Optimizer <T> {
	T		__eta;
	T		__beta1;
	T		__beta2;

	Matrix <T> *	__M = nullptr;
	Matrix <T> *	__Mh = nullptr;
	Matrix <T> *	__S = nullptr;
	Matrix <T> *	__Sh = nullptr;

	size_t		__size = 0;
	size_t		__iter = 0;
public:
	Adam(T eta = 0.001, T beta1 = 0.9, T beta2 = 0.999) : __eta(eta),
			__beta1(beta1), __beta2(beta2) {}

	~Adam() {
		delete[] __M;
		delete[] __Mh;
		delete[] __S;
		delete[] __Sh;
	}
	
	// Later reset if the layers pointer is different
	// from the cashed pointer
	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];
    		
		Matrix <T> *J = simple_gradient(layers, size, a, z, in, out, cost);

		if (size != __size)
			resize(size);

		for (size_t i = 0; i < __size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			// Resize the matrices if necessary
			resize(odim, i);

			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			__M[i] = __beta1 * __M[i] + (1 - __beta1) * J[i];
			__S[i] = __beta2 * __S[i] + (1 - __beta2) * shur(J[i], J[i]);

			__Mh[i] = __M[i] / T(1 + pow(__beta1, __iter));
			__Sh[i] = __S[i] / T(1 + pow(__beta2, __iter));

			J[i] = inv_shur(__eta * __Mh[i], __Sh[i] + e);
		}

		__iter++;

		delete[] a;
		delete[] z;

		return J;
	}
	
	void reset() {
		delete[] __M;
		delete[] __Mh;
		delete[] __S;
		delete[] __Sh;

		__size = 0;
		__iter = 0;

		__M = nullptr;
		__Mh = nullptr;
		__S = nullptr;
		__Sh = nullptr;
	}

	void resize(size_t size) {
		reset();

		__size = size;

		__S = new Matrix <T> [__size];
		__Sh = new Matrix <T> [__size];
		__M = new Matrix <T> [__size];
		__Mh = new Matrix <T> [__size];
	}

	inline void resize(const std::pair <size_t, size_t> &odim, size_t i) {
		if (__M[i].get_dimensions() != odim)
			__M[i] = Matrix <T> (odim.first, odim.second, T(0));
		if (__Mh[i].get_dimensions() != odim)
			__Mh[i] = Matrix <T> (odim.first, odim.second, T(0));
		if (__S[i].get_dimensions() != odim)
			__S[i] = Matrix <T> (odim.first, odim.second, T(0));
		if (__Sh[i].get_dimensions() != odim)
			__Sh[i] = Matrix <T> (odim.first, odim.second, T(0));
	}

	static const T epsilon;
};

template <class T>
const T Adam <T> ::epsilon = 1e-10;

}

}

#endif
