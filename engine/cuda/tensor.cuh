#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#define ZHP_CUDA

#include <tensor.hpp>

// CUDA headers
#include <cuda/error.cuh>

namespace zhetapi {

template <class T>
__host__ __device__
Tensor <T> ::Tensor() {}

template <class T>
__host__ __device__
Tensor <T> ::Tensor(const Tensor <T> &other) : __dims(other.__dims),
		__size(other.__size)
{
	__dim = new size_t[__dims];
	for (size_t i = 0; i < __dims; i++)
		__dim[i] = other.__dim[i];

	__array = new T[__size];
	for (size_t i = 0; i < __size; i++)
		__array[i] = other.__array[i];
}

template <class T>
__host__ __device__
Tensor <T> ::Tensor(size_t rows, size_t cols, const T &def)
{
	__dims = 2;
	__size = rows * cols;

	__dim = new size_t[2];
	
	__dim[0] = rows;
	__dim[1] = cols;

	__array = new T[__size];
	for (size_t i = 0; i < __size; i++)
		__array[i] = def;	
}

template <class T>
__host__ __device__
Tensor <T> ::Tensor(const ::std::vector <T> &arr) : __dims(1),
	__size(arr.size())
{
	__dim = new size_t[1];

	__dim[0] = __size;

	// TODO: Replace this exception handling
	// if (!__size)
	//	throw bad_dimensions();

	__array = new T[__size];

	for (size_t i = 0; i < __size; i++)
		__array[i] = arr[i];
}

template <class T>
__host__ __device__
Tensor <T> ::Tensor(const ::std::vector <size_t> &dim, const T &def)
{

#ifndef __CUDA_ARCH__

	__dims = dim.size();
	
	__dim = new size_t[__dims];

	size_t prod = 1;
	for (size_t i = 0; i < __dims; i++) {
		prod *= dim[i];

		__dim[i] = dim[i];
	}

	__size = prod;
	
	// TODO: Replace this exception handling
	// if (!__size)
	//	throw bad_dimensions();

	__array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		__array[i] = def;

#else

	__size = 0;
	__array = nullptr;
	
	__dims = 0;
	__dim = nullptr;

#endif

}

template <class T>
__host__ __device__
Tensor <T> ::~Tensor()
{
	clear();
}

template <class T>
__host__ __device__
void Tensor <T> ::clear(int id)
{
	// Skip if neither are allocated
	if (!this->__array && !this->__dim)
		return;

#ifndef __CUDA_ARCH__


	if (__on_device) {
		cudaFree(__dim);
		cudaCheckError(__dim);

		if (!__sliced) {
			cudaFree(__array);
			cudaCheckError(__array);
		}
	} else {
		delete[] __dim;

		if (!__sliced)
			delete[] __array;
	}
	
	__on_device = false;
	__sliced = false;

#else

	delete[] __dim;

	if (!__sliced)
		delete[] __array;

#endif

}


template <class T>
__host__ __device__
Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
{
	if (this != &other) {
		__dims = other.__dims;
		__size = other.__size;
	
		__dim = new size_t[__dims];
		for (size_t i = 0; i < __dims; i++)
			__dim[i] = other.__dim[i];

		__array = new T[__size];
		for (size_t i = 0; i < __size; i++)
			__array[i] = other.__array[i];
	}

	return *this;
}

template <class T>
__host__ __device__
bool operator==(const Tensor <T> &a, const Tensor <T> &b)
{
	if (a.__size != b.__size)
		return false;

	for (size_t i = 0; i < a.__size; i++) {
		if (a.__array[i] != b.__array[i])
			return false;
	}

	return true;
}

}

#endif
