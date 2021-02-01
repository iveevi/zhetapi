#ifndef STD_OPTIMIZERS_H_
#define STD_OPTIMIZERS_H_

// Engine headers
#include <optimizer.hpp>

namespace zhetapi {

namespace ml {

// TODO: Cache a and z vectors in the class itself
// In general cache as much data as possible
// Instead of creating a new gradient, modify the one passed

/*
 * This header contains the standard optimizers.
 *
 * Preferably use one optimizer per neural network.
 */

template <class T>
class SGD : public Optimizer <T> {
	T	__eta;
public:
	SGD(T eta = 0.001) : __eta(eta) {}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		for (size_t i = 0; i < size; i++)
			J[i] *= -1 * __eta;
		
		return J;
	}
};

/*
 * TODO: Cache J in the optimizer itself so that we do not need to repeatedly
 * allocate and free the gradient.
 */
template <class T>
class Momentum : public Optimizer <T> {
	T		__eta	= 0;
	T		__mu	= 0;

	Matrix <T> *	__M	= nullptr;
public:
	Momentum(T eta = 0.001, T mu = 0.9) : __eta(eta), __mu(mu) {}

	~Momentum() {
		delete[] __M;
	}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		if (this->__switch) {
			delete[] __M;

			__M = new Matrix <T> [size];
		}
		
		Matrix <T> *Jo = new Matrix <T> [size];
		for (size_t i = 0; i < size; i++) {
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (__M[i].get_dimensions() != odim)
				__M[i] = Matrix <T> (odim.first, odim.second, T(0));
			
			Jo[i] = __M[i] = __mu * __M[i] - __eta * J[i];
		}

		delete[] J;

		return Jo;
	}
};

// TODO: Inherit Nesterov from Momentum
template <class T>
class Nesterov : public Optimizer <T> {
	T		__eta;
	T		__mu;

	Matrix <T> *	__M = nullptr;
public:
	Nesterov(T eta = 0.001, T mu = 0.9) : __eta(eta), __mu(mu) {}

	~Nesterov() {
		delete[] __M;
	}

	Layer <T> *adjusted(Layer <T> *layers, size_t size)
	{
		Layer <T> *adj = new Layer <T> [size];
		
		for (size_t i = 0; i < size; i++)
			adj[i] = layers[i];

		if (size != this->__size)
			return adj;

		for (size_t i = 0; i < size; i++) {
			size_t rs = layers[i].get_fan_out();
			size_t cs = layers[i].get_fan_in() + 1;

			if (__M[i].get_dimensions() != std::make_pair(rs, cs))
				__M[i] = Matrix <T> (rs, cs, T(0));

			adj[i] += __mu * __M[i];
		}

		return adj;
	}

	Matrix <T> *raw_gradient(
				Layer <T> *layers,
				size_t size,
				const Vector <T> &in,
				const Vector <T> &out,
				Erf <T> *cost) override
	{
		Layer <T> *adj = adjusted(layers, size);

		Matrix <T> *Delta = Optimizer <T> ::raw_gradient(
				adj,
				size,
				in,
				out,
				cost);

		delete[] adj;

		return Delta;
	}

	Matrix <T> *raw_batch_gradient(
				Layer <T> *layers,
				size_t size,
				const DataSet <T> &ins,
				const DataSet <T> &outs,
				Erf <T> *cost) override
	{
		Layer <T> *adj = adjusted(layers, size);

		Matrix <T> *Delta = Optimizer <T> ::raw_batch_gradient(
				adj,
				size,
				ins,
				outs,
				cost);

		delete[] adj;

		return Delta;
	}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		if (this->__switch) {
			delete[] __M;

			__M = new Matrix <T> [size];
		}
		
		Matrix <T> *Jo = new Matrix <T> [size];
		for (size_t i = 0; i < size; i++) {
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (__M[i].get_dimensions() != odim)
				__M[i] = Matrix <T> (odim.first, odim.second, T(0));
			
			Jo[i] = __M[i] = __mu * __M[i] - __eta * J[i];
		}

		delete[] J;

		return Jo;
	}
};

template <class T>
class AdaGrad : public Optimizer <T> {
	T		__eta;

	Matrix <T> *	__S = nullptr;
public:
	AdaGrad(T eta = 0.001) : __eta(eta) {}

	~AdaGrad() {
		delete[] __S;
	}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		if (this->__switch) {
			delete[] __S;

			__S = new Matrix <T> [size];
		}

		Matrix <T> tmp;
		for (size_t i = 0; i < size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (__S[i].get_dimensions() != odim)
				__S[i] = Matrix <T> (odim.first, odim.second, T(0));

			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			__S[i] = __S[i] + shur(J[i], J[i]);

			tmp = __S[i] + e;

			tmp.pow(0.5);

			J[i] = inv_shur((-1 * __eta) * J[i], tmp);
		}

		return J;
	}

	static const T epsilon;
};

template <class T>
const T AdaGrad <T> ::epsilon = 1e-10;

template <class T>
class RMSProp : public Optimizer <T> {
	T		__eta;
	T		__beta;

	Matrix <T> *	__S = nullptr;
public:
	RMSProp(T eta = 0.001, T beta = 0.9) : __eta(eta), __beta(beta) {}

	~RMSProp() {
		delete[] __S;
	}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		if (this->__switch) {
			delete[] __S;

			__S = new Matrix <T> [size];
		}

		Matrix <T> tmp;
		for (size_t i = 0; i < size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (__S[i].get_dimensions() != odim)
				__S[i] = Matrix <T> (odim.first, odim.second, T(0));

			// TODO: Replace with element wise addition operation
			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			__S[i] = __beta * __S[i] + (1 - __beta) * shur(J[i], J[i]);

			tmp = __S[i] + e;

			tmp.pow(0.5);

			J[i] = inv_shur((-1 * __eta) * J[i], tmp);
		}

		return J;
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

	Matrix <T> *	__M	= nullptr;
	Matrix <T> *	__Mh	= nullptr;
	Matrix <T> *	__S	= nullptr;
	Matrix <T> *	__Sh	= nullptr;

	size_t		__iter	= 1;
public:
	Adam(T eta = 0.001, T beta1 = 0.9, T beta2 = 0.999)
			: __eta(eta), __beta1(beta1), __beta2(beta2) {}

	~Adam() {
		delete[] __M;
		delete[] __Mh;
		delete[] __S;
		delete[] __Sh;
	}
	
	// TODO: reset if the layers pointer is different
	// from the cashed pointer
	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		using namespace std;
		if (this->__switch)
			reset(size);

		Matrix <T> tmp;
		for (size_t i = 0; i < size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			// Resize the matrices if necessary
			resize(odim, i);

			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			__M[i] = __beta1 * __M[i] - (1 - __beta1) * J[i];
			__S[i] = __beta2 * __S[i] + (1 - __beta2) * shur(J[i], J[i]);

			__Mh[i] = __M[i] / T(1 - pow(__beta1, __iter));
			__Sh[i] = __S[i] / T(1 - pow(__beta2, __iter));

			tmp = __Sh[i] + e;

			tmp.pow(0.5);

			J[i] = inv_shur(__eta * __Mh[i], tmp);
		}

		__iter++;

		return J;
	}
	
	void reset(size_t size) {
		delete[] __M;
		delete[] __Mh;
		delete[] __S;
		delete[] __Sh;

		__iter = 1;

		__M = new Matrix <T> [size];
		__Mh = new Matrix <T> [size];
		__S = new Matrix <T> [size];
		__Sh = new Matrix <T> [size];
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
