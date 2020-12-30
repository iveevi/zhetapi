#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

// C/C++ headers
#include <algorithm>
#include <functional>

// Engine headers
#ifdef ZHP_CUDA

#include <cuda/vector.cuh>

#else

#include <vector.hpp>

#endif

#include <cuda/essentials.cuh>

namespace zhetapi {

namespace ml {

/*
* Scalar activation
*
* @tparam T the input and output type of the activation
*/
template <class T>
class Activation {
public:
	// TODO: Replace with a string
	enum activation_type {
		AT_Default,
		AT_Linear,
		AT_ReLU,
		AT_Sigmoid
	};

	// TODO: Add a vector <double> constructor for JSON
	__cuda_dual_prefix
	Activation();
	
	__cuda_dual_prefix
	Vector <T> compute(const Vector <T> &) const;

	__cuda_dual_prefix
	virtual Vector <T> operator()(const Vector <T> &) const;

	__cuda_dual_prefix
	virtual Activation *derivative() const;
	
	__cuda_dual_prefix
	int get_activation_type() const;

	template <class U>
	__cuda_dual_prefix
	friend Activation <U> *copy(Activation <U> *);
protected:
	activation_type kind;
};

#ifndef ZHP_CUDA

template <class T>
Activation <T> ::Activation() : kind(AT_Default) {}

// TODO: Reverse compute and operator()
template <class T>
Vector <T> Activation <T> ::operator()(const Vector <T> &x) const
{
	return x;
}

template <class T>
Vector <T> Activation <T> ::compute(const Vector <T> &x) const
{
	return (*this)(x);
}

template <class T>
Activation <T> *Activation <T> ::derivative() const
{
	return new Activation();
}

template <class T>
int Activation <T> ::get_activation_type() const
{
	return kind;
}

#endif

}

}

#endif
