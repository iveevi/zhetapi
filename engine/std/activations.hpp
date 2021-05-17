#ifndef STD_ACTIVATIONS_H_
#define STD_ACTIVATIONS_H_

// Engine headers
#ifdef ZHP_CUDA

#include "../cuda/activation.cuh"

#else

#include "../activation.hpp"
#include "activation_derivatives.hpp"

#endif

// Engine standard headers
#include "loaders.hpp"

// Engine CUDA headers
#include "../cuda/essentials.cuh"

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
	T	_alpha;
public:
	
	__cuda_dual__
	Linear(const T &alpha = T(1)) : _alpha(alpha),
			Activation <T> (Activation <T> ::AT_Linear, {alpha}) {}

	__cuda_dual__
	Activation <T> *copy() const {
		return new Linear <T> (_alpha);
	}

#ifndef ZHP_CUDA

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				return x[i] * _alpha;
			}
		);
	}

	Activation <T> *derivative() const {
		return new _DLinear <T> (_alpha);
	}

#else
	
	_host_ _device_
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x, this] _host_ _device_ (size_t i) {
				return x[i] * _alpha;
			}
		);
	}

	_host_ _device_
	Activation <T> *derivative() const {
		return new _DLinear <T> (_alpha);
	}

#endif

};

template <class T>
class ReLU : public Activation <T> {
public:

#ifdef __AVR

	__cuda_dual__
	ReLU() : Activation <T> (Activation <T> ::AT_ReLU) {}

#else

	__cuda_dual__
	ReLU() : Activation <T> (Activation <T> ::AT_ReLU, {}) {}

#endif

	__cuda_dual__
	Activation <T> *copy() const {
		return new ReLU();
	}

// TODO: reverse the order
#ifndef ZHP_CUDA

	Vector <T> compute(const Vector <T> &x) const {
		T *arr = new T[x.size()];
		for (size_t i = 0; i < x.size(); i++) {
			arr[i] = (x[i] > 0) ? x[i] : 0;
		}
		return Vector <T> (x.size(), arr, false);
	}

	Activation <T> *derivative() const {
		return new _DReLU <T> ();
	}

#else
	
	_host_ _device_
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x] _host_ _device_ (size_t i) {
				return (x[i] > 0) ? x[i] : 0;
			}
		);
	}

	_host_ _device_
	Activation <T> *derivative() const {
		return new _DReLU <T> ();
	}

#endif

};

template <class T>
class LeakyReLU : public Activation <T> {
	T	_alpha;
public:
	__cuda_dual__
	LeakyReLU(const T &alpha = 1) :
			Activation <T> (Activation <T> ::AT_ReLU, {alpha}),
			_alpha(alpha) {}
	
	__cuda_dual__
	Activation <T> *copy() const {
		return new LeakyReLU <T> (_alpha);
	}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				return (x[i] > 0) ? x[i] : _alpha * x[i];
			}
		);
	}

	Activation <T> *derivative() const {
		return new _DLeakyReLU <T> (_alpha);
	}
};

template <class T>
class Sigmoid : public Activation <T> {
public:

	__cuda_dual__
	Sigmoid() : Activation <T> (Activation <T> ::AT_Sigmoid, {}) {}

	__cuda_dual__
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
		return new _DSigmoid <T> ();
	}

#else

	_host_ _device_
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x] _host_ _device_ (size_t i) { 
				return 1/(1 + exp(-x[i]));
			}
		);
	}

	_host_ _device_
	Activation <T> *derivative() const {
		return new _DSigmoid <T> ();
	}

#endif

};

template <class T>
class ScaledSigmoid : public Activation <T> {
	T	_alpha;
public:
	ScaledSigmoid(const T &alpha) : _alpha(alpha) {}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(), [&](size_t i) {return
				_scaled_sigmoid(x[i], _alpha);});
	}

	Activation <T> *derivative() const {
		return new _DScaledSigmoid <T> (_alpha);
	}
};

template <class T>
class Softmax : public Activation <T> {
public:
	Softmax() : Activation <T> ({}) {}

	__cuda_dual__
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
		return new _DSoftmax <T> ();
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
		return new _DSoftmax <T> ();
	}
};

#ifndef __AVR

template <class T>
void ZhetapiInit()
{
	_zhp_register_activation(Linear, T, load_linear <T>);
	_zhp_register_activation(ReLU, T, load_relu <T>);
	_zhp_register_activation(Sigmoid, T, load_sigmoid <T>);
	_zhp_register_activation(Softmax, T, load_softmax <T>);
}

#endif

// TODO: Make a more generalized version,
// which will also work on the gpu (the issues is virtual
// functions; maybe returning a _host_ _device_
// function pointer for computation will work)
template <class T>
__cuda_dual__
Activation <T> *copy(Activation <T> *act)
{
	switch (act->_kind) {
	case Activation <T> ::AT_Default:
		return new Activation <T> ();
	case Activation <T> ::AT_Linear:
		// TODO: Try to transfer _alpha content
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
