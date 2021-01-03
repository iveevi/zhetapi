#ifndef ERF_H_
#define ERF_H_

// C/C++ headers
#include <cmath>
#include <memory>

// Engine headers
#ifdef ZHP_CUDA

#include <cuda/vector.cuh>

#else

#include <vector.hpp>

#endif

#include <cuda/essentials.cuh>

namespace zhetapi {
		
namespace ml {

template <class T>
class Erf {
public:
	// TODO: Replace with a string
	enum erf_type {
		OPT_Default,
		OPT_SE,
		OPT_MSE,
	};

	// TODO: Add a vector <double> constructor for JSON
	__cuda_dual_prefix
	Erf();

	__cuda_dual_prefix
	Vector <T> compute(const Vector <T> &, const Vector <T> &) const;

	__cuda_dual_prefix
	virtual Vector <T> operator()(const Vector <T> &, const Vector <T> &) const;

	__cuda_dual_prefix
	virtual Erf *derivative() const;

	__cuda_dual_prefix
	int get_erf_type() const;

	template <class U>
	__cuda_dual_prefix
	friend Erf <U> *copy(Erf <U> *);
protected:
	erf_type kind;
};

#ifndef ZHP_CUDA

template <class T>
Erf <T> ::Erf() : kind(OPT_Default) {}

// TODO: Reverse compute and operator()
template <class T>
Vector <T> Erf <T> ::operator()(const Vector <T> &comp, const Vector <T> &in) const
{
	return {(comp - in).norm()};
}

template <class T>
Vector <T> Erf <T> ::compute(const Vector <T> &comp, const Vector <T> &in) const
{
	return (*this)(comp, in);
}

template <class T>
Erf <T> *Erf <T> ::derivative() const
{
	return new Erf();
}

template <class T>
int Erf <T> ::get_erf_type() const
{
	return kind;
}

#endif

}

}

#endif
