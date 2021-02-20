#ifndef TENSOR_H_
#define TENSOR_H_

// C/C++ headers
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <cuda/essentials.cuh>

namespace zhetapi {

// Forward declarations
template <class T>
class Tensor;

// Tensor_type operations
template <class T>
struct Tensor_type : std::false_type {};

template <class T>
struct Tensor_type <Tensor <T>> : std::true_type {};

template <class T>
bool is_tensor_type()
{
	return Tensor_type <T> ::value;
}

// Tensor class
template <class T>
class Tensor {
protected:
	T	*__array = nullptr;
	size_t	__size = 0;

	// Variables for printing
	size_t	*__dim = nullptr;
	size_t	__dims = 0;

	bool	__sliced = false;	// Flag for no deallocation

#ifdef ZHP_CUDA

	bool	__on_device = false;	// Flag for device allocation

#endif

public:
	Tensor(const std::vector <std::size_t> &, const ::std::vector <T> &);

	// TODO: remove size term from vector and matrix classes
	size_t size() const;
	
	__cuda_dual_prefix
	void clear();

	bool good() const;

	// Boolean operators (generalize with prefix)
	template <class U>
	friend bool operator==(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend bool operator!=(const Tensor <U> &, const Tensor <U> &);
	
	// Printing functions
	std::string print() const;

	template <class U>
	friend std::ostream &operator<<(std::ostream &, const Tensor <U> &);

	// Dimension mismatch exception
	class dimension_mismatch {};
	class bad_dimensions {};

#ifndef ZHP_CUDA

	// Construction and memory
	Tensor();
	Tensor(const Tensor &);
	Tensor(const std::vector <std::size_t> &);
	Tensor(const std::vector <std::size_t> &, const T &);

	// Cross-type operations
	template <class A>
	std::enable_if <is_tensor_type <A> (), Tensor &> operator=(const Tensor <A> &);

	~Tensor();

	Tensor &operator=(const Tensor &);

	// Indexing
	T &operator[](const std::vector <size_t> &);
	const T &operator[](const std::vector <size_t> &) const;

#else

	__host__ __device__
	Tensor();

	__host__ __device__
	Tensor(bool);

	__host__ __device__
	Tensor(const Tensor &);
	
	__host__ __device__
	Tensor(size_t, size_t, const T &);

	__host__ __device__
	Tensor(const std::vector <T> &);

	__host__ __device__
	Tensor(const std::vector <std::size_t> &, const T & = T());

	__host__ __device__
	~Tensor();

	__host__ __device__
	Tensor &operator=(const Tensor &);

	template <class U>
	__host__ __device__
	friend bool operator==(const Tensor <U> &, const Tensor <U> &);

#endif

};

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim, const std::vector <T> &arr)
		: __dims(dim.size())
{

#ifdef ZHP_CUDA

	__on_device = false;

#endif

	__dim = new size_t[__dims];

	size_t prod = 1;
	for (size_t i = 0; i < __dims; i++) {
		prod *= dim[i];

		__dim[i] = dim[i];
	}

	__size = prod;

	if (__size <= 0)
		throw bad_dimensions();

	if (arr.size() != __size)
		throw dimension_mismatch();

	__array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		__array[i] = arr[i];
}

template <class T>
size_t Tensor <T> ::size() const
{
	return __size;
}

// Boolean comparison
template <class T>
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

template <class T>
bool operator!=(const Tensor <T> &a, const Tensor <T> &b)
{
	return !(a == b);
}

// Printing functions
template <class T>
std::string print(T *arr, size_t size, size_t *ds, size_t dn, size_t dmax)
{
	if (size == 0)
		return "[]";
	
	std::string out = "[";

	// Size of each dimension
	size_t dsize = size / ds[dn];

	T *current = arr;
	for (size_t i = 0; i < ds[dn]; i++) {
		if (dn == dmax)
			out += std::to_string(*current);
		else
			out += print(current, dsize, ds, dn + 1, dmax);

		if (i < ds[dn] - 1)
			out += ", ";

		current += dsize;
	}

	return out + "]";
}

template <class T>
std::ostream &operator<<(std::ostream &os, const Tensor <T> &ts)
{
	os << print(ts.__array, ts.__size, ts.__dim, 0, ts.__dims - 1);

	return os;
}

#ifndef ZHP_CUDA

// Constructors and memory relevant functions
template <class T>
Tensor <T> ::Tensor() : __dim(nullptr), __dims(0), __array(nullptr),
		__size(0) {}

template <class T>
Tensor <T> ::Tensor(const Tensor <T> &other) : __dims(other.__dims), __size(other.__size)
{
	__dim = new size_t[__dims];
	for (size_t i = 0; i < __dims; i++)
		__dim[i] = other.__dim[i];

	__array = new T[__size];
	for (size_t i = 0; i < __size; i++)
		__array[i] = other.__array[i];
}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim)
		: __dims(dim.size())
{
	__dim = new size_t[__dims];

	size_t prod = 1;
	for (size_t i = 0; i < __dims; i++) {
		prod *= dim[i];

		__dim[i] = dim[i];
	}

	__size = prod;

	if (!__size)
		return;

	__array = new T[prod];
}

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

	__size = prod;

	if (!__size)
		return;

	__array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		__array[i] = def;
}

template <class T>
Tensor <T> ::~Tensor()
{
	clear();
}

template <class T>
void Tensor <T> ::clear()
{
	if (!__array && !__dim)
		return;

	if (__dim)
		delete[] __dim;

	if (__array && !__sliced)
		delete[] __array;

	__array = nullptr;
	__dim = nullptr;
}

template <class T>
bool Tensor <T> ::good() const
{
	return __array != nullptr;
}

template <class T>
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

// Index
template <class T>
T &Tensor <T> ::operator[](const ::std::vector <size_t> &indices)
{
	size_t full = 0;

	assert(indices.size() == __dims);
	for (size_t i = 0; i < __dims; i++)
		full += indices[i] * __dim[__dims - (i + 1)];
	
	return __array[full];
}

template <class T>
const T &Tensor <T> ::operator[](const ::std::vector <size_t> &indices) const
{
	size_t full = 0;

	assert(indices.size() == __dims);
	for (size_t i = 0; i < __dims; i++)
		full += indices[i] * __dim[__dims - (i + 1)];
	
	return __array[full];
}

#endif

}

#endif
