#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

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
class Optimizer {
public:
	// TODO: Replace with a string
	enum optimizer_type {
		OPT_Default,
		OPT_SE,
		OPT_MSE,
	};

	// TODO: Add a vector <double> constructor for JSON
	__cuda_dual_prefix
	Optimizer();

	__cuda_dual_prefix
	Vector <T> compute(const Vector <T> &, const Vector <T> &) const;

	__cuda_dual_prefix
	virtual Vector <T> operator()(const Vector <T> &, const Vector <T> &) const;

	__cuda_dual_prefix
	virtual Optimizer *derivative() const;

	__cuda_dual_prefix
	int get_optimizer_type() const;

	template <class U>
	__cuda_dual_prefix
	friend Optimizer <U> *copy(Optimizer <U> *);
protected:
	optimizer_type kind;
};

#ifndef ZHP_CUDA

template <class T>
Optimizer <T> ::Optimizer() : kind(OPT_Default) {}

// TODO: Reverse compute and operator()
template <class T>
Vector <T> Optimizer <T> ::operator()(const Vector <T> &comp, const Vector <T> &in) const
{
	return {(comp - in).norm()};
}

template <class T>
Vector <T> Optimizer <T> ::compute(const Vector <T> &comp, const Vector <T> &in) const
{
	return (*this)(comp, in);
}

template <class T>
Optimizer <T> *Optimizer <T> ::derivative() const
{
	return new Optimizer();
}

template <class T>
int Optimizer <T> ::get_optimizer_type() const
{
	return kind;
}

#endif

}

}

#endif
