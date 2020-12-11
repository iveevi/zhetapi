#ifndef MATRIX_CUH_
#define MATRIX_CUH_

#define ZHP_CUDA

#include <vector.hpp>

namespace zhetapi {
	
	// Indexing operators
	template <class T>
	__host__ __device__
	T &Vector <T> ::operator[](size_t i)
	{
		return this->__array[i];
	}

	template <class T>
	__host__ __device__
	const T &Vector <T> ::operator[](size_t i) const
	{
		return this->__array[i];
	}

	// Miscellaneous functions
	template <class T>
	T Vector <T> ::norm() const
	{
		return sqrt(inner(*this, *this));
	}

}

#endif
