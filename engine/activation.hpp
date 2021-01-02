#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

// C/C++ headers
#include <algorithm>
#include <functional>
#include <map>

// Engine headers
#ifdef ZHP_CUDA

#include <cuda/vector.cuh>

#else

#include <vector.hpp>

#endif

#include <cuda/essentials.cuh>

// A is class name, T is the type (template), L is the loader function
#define __zhp_register_activation(A, T, L)		\
	zhetapi::ml::Activation <T>			\
	::__activation_loaders[typeid(A <T>).name()] = L;

namespace zhetapi {

namespace ml {

template <class T>
class Activation;

// Format of an activation loader
template <class T>
using loader = Activation <T> *(*)(const std::vector <double> &);

/*
 * Represents an activation in machine learning. Takes a vector of type T as an
 * input and returns a vector of type T.
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

	// Global list of all registered activations
	static std::map <std::string, loader <T>> __activation_loaders;

	static void display_loaders()
	{
		using namespace std;
		cout << "LOADERS:" << endl;
		for (auto pr : __activation_loaders)
			cout << pr.first << " is registered" << endl;
	}
protected:
	activation_type kind;
};


template <class T>
std::map <std::string, loader <T>> Activation <T> ::__activation_loaders;

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
