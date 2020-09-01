#ifndef TENSOR_H_
#define TENSOR_H_

#include <cstdlib>
#include <string>
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

	// Printing functions
	std::string print() const;
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

template <class T>
std::string Tensor <T> ::print() const
{
	if (__dims == 0)
		return std::to_string(__array[0]);
	
	std::string out = "[";

	std::vector <size_t> cropped;

	using namespace std;
	for (int i = 0; i < ((int) __dims) - 1; i++)
		cropped.push_back(__dim[i + 1]);
	
	for (size_t i = 0; i < __dim[0]; i++) {
		Tensor tmp(cropped, __array[0]);

		out += tmp.print();

		if (i < __dim[0] - 1)
			out += ",";
	}

	return out + "]";
}

#endif