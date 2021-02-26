#ifndef STD_ACTIVATIONS_H_
#define STD_ACTIVATIONS_H_

// Engine headers
#ifdef ZHP_CUDA

#include <cuda/activation.cuh>

#else

#include <activation.hpp>

#include <core/activation_derivatives.hpp>

#endif

// Engine standard headers
#include <std/loaders.hpp>

// Engine CUDA headers
#include <cuda/essentials.cuh>

namespace zhetapi {
		
namespace ml {

/*
* All activation classes have inlined member functions as the
* operations they perform are very minimal. These activation classes
* serve only as a wrapper for their underlying functions as well as a
* means of extracting their derivatives.
*/

// Linear activation class
template <class T>
class Linear : public Activation <T> {
	T	__alpha;
public:
	
	__cuda_dual_prefix
	Linear(const T &alpha = T(1)) : __alpha(alpha),
			Activation <T> (Activation <T> ::AT_Linear, {alpha}) {}

	__cuda_dual_prefix
	Activation <T> *copy() const {
		return new Linear <T> (__alpha);
	}

#ifndef ZHP_CUDA

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				return x[i] * __alpha;
			}
		);
	}

	Activation <T> *derivative() const {
		return new __DLinear <T> (__alpha);
	}

#else
	
	__host__ __device__
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x, this] __host__ __device__ (size_t i) {
				return x[i] * __alpha;
			}
		);
	}

	__host__ __device__
	Activation <T> *derivative() const {
		return new __DLinear <T> (__alpha);
	}

#endif

};

template <class T>
class ReLU : public Activation <T> {
public:

	__cuda_dual_prefix
	ReLU() : Activation <T> (Activation <T> ::AT_ReLU, {}) {}

	__cuda_dual_prefix
	Activation <T> *copy() const {
		return new ReLU();
	}

#ifndef ZHP_CUDA

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				return (x[i] > 0) ? x[i] : 0;
			}
		);
	}

	Activation <T> *derivative() const {
		return new __DReLU <T> ();
	}

#else
	
	__host__ __device__
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x] __host__ __device__ (size_t i) {
				return (x[i] > 0) ? x[i] : 0;
			}
		);
	}

	__host__ __device__
	Activation <T> *derivative() const {
		return new __DReLU <T> ();
	}

#endif

};

template <class T>
class LeakyReLU : public Activation <T> {
	T	__alpha;
public:
	__cuda_dual_prefix
	LeakyReLU(const T &alpha = 1) :
			Activation <T> (Activation <T> ::AT_ReLU, {alpha}) {}
	
	__cuda_dual_prefix
	Activation <T> *copy() const {
		return new LeakyReLU <T> (__alpha);
	}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				return (x[i] > 0) ? x[i] : __alpha * x[i];
			}
		);
	}

	Activation <T> *derivative() const {
		return new __DLeakyReLU <T> (__alpha);
	}
};

template <class T>
class Sigmoid : public Activation <T> {
public:

	__cuda_dual_prefix
	Sigmoid() : Activation <T> (Activation <T> ::AT_Sigmoid, {}) {}

	__cuda_dual_prefix
	Activation <T> *copy() const {
		return new Sigmoid();
	}

#ifndef ZHP_CUDA

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				return 1/(1 + exp(-x[i]));
			}
		);
	}

	Activation <T> *derivative() const {
		return new __DSigmoid <T> ();
	}

#else

	__host__ __device__
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x] __host__ __device__ (size_t i) { 
				return 1/(1 + exp(-x[i]));
			}
		);
	}

	__host__ __device__
	Activation <T> *derivative() const {
		return new __DSigmoid <T> ();
	}

#endif

};

template <class T>
class ScaledSigmoid : public Activation <T> {
	T	__alpha;
public:
	ScaledSigmoid(const T &alpha) : __alpha(alpha) {}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(), [&](size_t i) {return
				__scaled_sigmoid(x[i], __alpha);});
	}

	Activation <T> *derivative() const {
		return new __DScaledSigmoid <T> (__alpha);
	}
};

template <class T>
class Softmax : public Activation <T> {
public:
	Softmax() : Activation <T> ({}) {}

	__cuda_dual_prefix
	Activation <T> *copy() const {
		return new Softmax();
	}

	Vector <T> compute(const Vector <T> &x) const {
		// Subtract by max for numerical stability
		T _max = x[0];
		for (size_t i = 1; i < x.size(); i++)
			_max = (_max > x[i]) ? _max : x[i];

		T _sum = 0;
		for (size_t i = 0; i < x.size(); i++)
			_sum += exp(x[i] - _max);

		return Vector <T> (x.size(),
			[&](size_t i) {
				return exp(x[i] - _max)/_sum;
			}
		);
	}

	Activation <T> *derivative() const {
		return new __DSoftmax <T> ();
	}
};

template <class T>
class SoftmaxInterval : public Activation <T> {
public:
	Vector <T> compute(const Vector <T> &x) const {
		// Subtract by max for numerical stability
		T _max = x[0];
		for (size_t i = 1; i < x.size(); i++)
			_max = (_max > x[i]) ? _max : x[i];

		T _sum;
		for (size_t i = 0; i < x.size(); i++)
			_sum += exp(x[i] - _max);

		T _acc = 0;
		return Vector <T> (x.size(),
			[&](size_t i) {
				_acc += exp(x[i] - _max)/_sum;

				return _acc;
			}
		);
	}

#warning "The derivative for the SoftmaxInterval activation has not yet been properly configured."

	Activation <T> *derivative() const {
		return new __DSoftmax <T> ();
	}
};

template <class T>
void ZhetapiInit()
{
	__zhp_register_activation(Linear, T, load_linear <T>);
	__zhp_register_activation(ReLU, T, load_relu <T>);
	__zhp_register_activation(Sigmoid, T, load_sigmoid <T>);
	__zhp_register_activation(Softmax, T, load_softmax <T>);
}

// TODO: Make a more generalized version,
// which will also work on the gpu (the issues is virtual
// functions; maybe returning a __host__ __device__
// function pointer for computation will work)
template <class T>
__cuda_dual_prefix
Activation <T> *copy(Activation <T> *act)
{
	switch (act->__kind) {
	case Activation <T> ::AT_Default:
		return new Activation <T> ();
	case Activation <T> ::AT_Linear:
		// TODO: Try to transfer __alpha content
		return new Linear <T> ();
	case Activation <T> ::AT_ReLU:
		return new ReLU <T> ();
	case Activation <T> ::AT_Sigmoid:
		return new Sigmoid <T> ();
	}

	return nullptr;
}

}

}

#endif
