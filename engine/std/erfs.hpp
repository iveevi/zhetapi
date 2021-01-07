#ifndef STD_ERFS_H_
#define STD_ERFS_H_

// Engine headers
#include <erf.hpp>

// Engine CUDA headers
#include <cuda/essentials.cuh>

namespace zhetapi {
		
namespace ml {
	
/*
* All Erf classes have inlined member functions for the same
* reason that the activation classes are inlined. Obscure naming is
* also done for the same reason.
*/

// Squared error
template <class T>
class __DSquaredError : public Erf <T> {
public:
	__cuda_dual_prefix
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		return -T(2) * (comp - in);
	}
};

template <class T>
class SquaredError : public Erf <T> {
public:
	__cuda_dual_prefix
	SquaredError() {
		this->kind = Erf <T> ::OPT_SE; 
	}

	__cuda_dual_prefix
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		T sum = 0;

		for (size_t i = 0; i < comp.size(); i++)
			sum += (comp[i] - in[i]) * (comp[i] - in[i]);
		
		return Vector <T> (1, sum);
	}

	__cuda_dual_prefix
	Erf <T> *derivative() const
	{
		return new __DSquaredError <T> ();
	}
};

// Mean squared error
template <class T>
class __DMeanSquaredError : public Erf <T> {
public:
	__cuda_dual_prefix
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		return -T(2)/T(comp.size()) * (comp - in);
	}
};

template <class T>
class MeanSquaredError : public Erf <T> {
public:
	__cuda_dual_prefix
	MeanSquaredError() {
		this->kind = Erf <T> ::OPT_MSE;
	}

	__cuda_dual_prefix
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		T sum = 0;

		for (size_t i = 0; i < comp.size(); i++)
			sum += (comp[i] - in[i]) * (comp[i] - in[i]);
		
		return Vector <T> (1, sum / T(comp.size()));
	}

	__cuda_dual_prefix
	Erf <T> *derivative() const {
		return new __DMeanSquaredError <T> ();
	}
};

// Copy base activations
template <class T>
__cuda_dual_prefix
Erf <T> *copy(Erf <T> *opt)
{
	switch (opt->kind) {
	case Erf <T> ::OPT_Default:
		return new Erf <T> ();
	case Erf <T> ::OPT_SE:
		return new SquaredError <T> ();
	case Erf <T> ::OPT_MSE:
		return new MeanSquaredError <T> ();
	}

	return nullptr;
}

}

}

#endif
