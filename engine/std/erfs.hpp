#ifndef STD_ERFS_H_
#define STD_ERFS_H_

// Engine headers
#include <erf.hpp>

#include <std/erf_derivatives.hpp>

// Engine CUDA headers
#include <cuda/essentials.cuh>

namespace zhetapi {
		
namespace ml {
	
/*
* All Erf classes have inlined member functions for the same
* reason that the activation classes are inlined. Obscure naming is
* also done for the same reason.
*/

template <class T>
class SE : public Erf <T> {
public:
	__cuda_dual__
	SE() {
		this->kind = Erf <T> ::OPT_SE; 
	}

	__cuda_dual__
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		Erf <T> ::assert_size(comp, in);
		
		T sum = 0;

		for (size_t i = 0; i < comp.size(); i++)
			sum += (comp[i] - in[i]) * (comp[i] - in[i]);
		
		return Vector <T> (1, sum);
	}

	__cuda_dual__
	Erf <T> *derivative() const
	{
		return new _DSE <T> ();
	}
};

template <class T>
class MSE : public Erf <T> {
public:
	__cuda_dual__
	MSE() {
		this->kind = Erf <T> ::OPT_MSE;
	}

	__cuda_dual__
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		Erf <T> ::assert_size(comp, in);
		
		T sum = 0;

		for (size_t i = 0; i < comp.size(); i++)
			sum += (comp[i] - in[i]) * (comp[i] - in[i]);
		
		return Vector <T> (1, sum / T(comp.size()));
	}

	__cuda_dual__
	Erf <T> *derivative() const {
		return new _DMSE <T> ();
	}
};

// Copy base activations
template <class T>
__cuda_dual__
Erf <T> *copy(Erf <T> *opt)
{
	switch (opt->kind) {
	case Erf <T> ::OPT_Default:
		return new Erf <T> ();
	case Erf <T> ::OPT_SE:
		return new SE <T> ();
	case Erf <T> ::OPT_MSE:
		return new MSE <T> ();
	}

	return nullptr;
}

}

}

#endif
