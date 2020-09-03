#ifndef STD_OPTIMIZER_FUNCTIONS_H_
#define STD_OPTIMIZER_FUNCTIONS_H_

// C/C++ headers
#include <cstdlib>

// Engine headers
#include <vector.hpp>

namespace ml {

	// Squared error
	template <class T>
	T __squared(const Vector <T> &comp, const Vector <T> &in)
	{
		T sum = 0;

		for (size_t i = 0; i < comp.size(); i++)
			sum += (comp[i] - in[i]) * (comp[i] - in[i]);
		
		return sum;
	}

	// Mean squared error
	template <class T>
	T __mean_squared(const Vector <T> &comp, const Vector <T> &in)
	{
		T sum = 0;

		for (size_t i = 0; i < comp.size(); i++)
			sum += (comp[i] - in[i]) * (comp[i] - in[i]);
		
		return sum/T(comp.size());
	}

}

#endif