#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// C/C++ headers
#include <cmath>

// Engine headers
#include <vector.hpp>

namespace ml {

	template <class T>
	class Optimizer {
	public:
		Optimizer();

		virtual T operator()(const Vector <T> &, const Vector <T> &) const;
	};

	template <class T>
	Optimizer <T> ::Optimizer() {}

	template <class T>
	T Optimizer <T> ::operator()(const Vector <T> &comp, const Vector <T> &in) const
	{
		return (comp - in).norm();
	}

}

#endif