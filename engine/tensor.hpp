#ifndef TENSOR_H_
#define TENSOR_H_

#include <cstdlib>
#include <vector>

template <class T>
class Tensor {
	size_t *	__dim;
	T *		__array;

	size_t		__dims;
public:
	Tensor();
	Tensor(const std::vector <std::size_t> &, const T & = T());

	~Tensor();
};

template <class T>
Tensor <T> ::Tensor() : __dim(nullptr), __array(nullptr) {}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim, const T &def)
		: __dims(dim.size())
{
	__dim = new size_t[__dims];

	size_t prod = 1;
	for (size_t i = 0; i < __dims; i++) {
		prod *= dim[i];

		__dim[i] = dim[i];
	}

	__array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		__array[i] = def;
}

template <class T>
Tensor <T> ::~Tensor()
{
	delete[] __dim;
	delete[] __array;
}

#endif