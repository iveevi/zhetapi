#ifndef TENSOR_H_
#define TENSOR_H_

// Standard headers
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// Library headers
#include "std/interval.hpp"
#include "cuda/essentials.cuh"

namespace zhetapi {

// Forward declarations
template <class T>
class Tensor;

template <class T>
class Matrix;

template <class T>
class Vector;

namespace utility {

template <size_t N>
class Interval;

}

// Tensor class
template <class T>
class Tensor {
protected:
	// TODO: dimension as a vector
	size_t	_dims		= 0;
	size_t *_dim		= nullptr;
	bool	_dim_sliced	= false;

	size_t	_size		= 0;
	T *	_array		= nullptr;
	bool	_arr_sliced	= false;

#ifdef __CUDACC__

	NVArena *_arena		= nullptr;
	bool	_on_device	= false;

#endif

public:
	// Public type aliases
	using value_type = T;
	using shape_type = std::vector <std::size_t>;

	// Essential constructors
	Tensor();
	Tensor(const Tensor &);

	template <class A>
	Tensor(const Tensor <A> &);

	// Single element
	Tensor(const T &);

	Tensor(size_t, size_t);
	Tensor(size_t, size_t *, size_t, T *, bool = true);

	explicit Tensor(const std::vector <std::size_t> &);
	Tensor(const std::vector <std::size_t> &, const T &);
	Tensor(const std::vector <std::size_t> &, const std::vector <T> &);

	// Indexing
	Tensor <T> operator[](size_t);

	// TODO: iterators
	// TODO: this type of indexing is very tedious with the [], use anotehr
	// method like .get(...)
	T &operator[](const std::vector <size_t> &);
	const T &operator[](const std::vector <size_t> &) const;

	// TODO: remove size term from vector and matrix classes
	__cuda_dual__ size_t size() const;
	// __cuda_dual__ size_t dimensions() const;
	__cuda_dual__ size_t dim_size(size_t) const;
	__cuda_dual__ size_t safe_dim_size(size_t) const;

	// TODO: refactor to shape()
	shape_type shape() const;

	// TODO: private?
	__cuda_dual__
	void clear();

	// Properties
	bool good() const;

	// Actions
	void nullify(long double, const utility::Interval <1> &);

	// Boolean operators (generalize with prefix)
	template <class U>
	friend bool operator==(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend bool operator!=(const Tensor <U> &, const Tensor <U> &);

	// Printing functions
	std::string print() const;

	template <class U>
	friend std::ostream &operator<<(std::ostream &, const Tensor <U> &);

	template <class A>
	Tensor &operator=(const Tensor <A> &);

	Tensor &operator=(const Tensor &);

	~Tensor();

	// TODO: Re-organize the methods
	Vector <T> cast_to_vector() const;
	Matrix <T> cast_to_matrix(size_t, size_t) const;

	// Arithmetic
	void operator*=(const T &);
	void operator/=(const T &);

	template <class U>
	friend Matrix <U> operator*(const Matrix <U> &, const U &);

	template <class U>
	friend Matrix <U> operator*(const U &, const Matrix <U> &);

	template <class U>
	friend Matrix <U> operator/(const Matrix <U> &, const U &);

	template <class U>
	friend Matrix <U> operator/(const U &, const Matrix <U> &);

	// Dimension mismatch exception
	class dimension_mismatch {};
	class bad_dimensions {};

#ifdef __CUDACC__

	class null_nvarena : public std::runtime_error {
	public:
		null_nvarena() : std::runtime_error("Tensor::clear: null NVArena.") {}
	};

#endif

	// TODO: start reworking tensor from here

	// Get scalar element
	const T &get(size_t) const;

	// Apply a function to each element of the tensor
	Tensor <T> transform(T (*)(const T &)) const;
	Tensor <T> transform(const std::function <T (const T &)> &) const;

	// Arithmetic
	// TODO: heterogneous versions?
	template <class U>
	friend Tensor <U> operator+(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend Tensor <U> operator-(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend Tensor <U> multiply(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend Tensor <U> divide(const Tensor <U> &, const Tensor <U> &);
};

// TODO: reorganize

/**
 * @brief Default constructor.
 */
template <class T>
Tensor <T> ::Tensor() {}

/**
 * @brief Homogenous (with respect to the component type) copy constructor.
 *
 * @param other the reference vector (to be copied from).
 */
template <class T>
Tensor <T> ::Tensor(const Tensor <T> &other)
		: _size(other._size), _dims(other._dims)
{
	// Faster for homogenous types
	_dim = new size_t[_dims];
	memcpy(_dim, other._dim, sizeof(size_t) * _dims);

	_array = new T[_size];
	memcpy(_array, other._array, sizeof(T) * _size);
}

/**
 * @brief Heterogenous (with respect to the component type) copy constructor.
 *
 * @param other the reference vector (to be copied from).
 */
template <class T>
template <class A>
Tensor <T> ::Tensor(const Tensor <A> &other)
		: _size(other._size), _dims(other._dims)
{
	_dim = new size_t[_dims];
	memcpy(_dim, other._dim, sizeof(size_t) * _dims);

	_array = new T[_size];
	for (size_t i = 0; i < _size; i++)
		_array[i] = static_cast <T> (other._array[i]);
}

// Single element
template <class T>
Tensor <T> ::Tensor(const T &value)
		: _size(1), _dims(1)
{
	_dim = new size_t[1];
	_dim[0] = 1;

	_array = new T[1];
	_array[0] = value;
}

template <class T>
Tensor <T> ::Tensor(size_t rows, size_t cols)
		: _dims(2), _size(rows * cols)
{
	_dim = new size_t[2];
	_dim[0] = rows;
	_dim[1] = cols;

	_array = new T[_size];
}

/**
 * @brief Full slice constructor. Makes a Tensor out of existing (previously
 * allocated memory), and the ownership can be decided. By default, the Tensor
 * does not gain ownership over the memory. Note that the sizes given are not
 * checked for validity.
 *
 * @param dims the number of dimensions to slice.
 * @param dim the dimension size array.
 * @param size the size of the Tensor.
 * @param array the components of the Tensor.
 * @param slice the slice flag. Set to \c true to make sure the memory is not
 * deallocated by the resulting Tensor, and \c false otherwise.
 */
template <class T>
Tensor <T> ::Tensor(size_t dims, size_t *dim, size_t size, T *array, bool slice)
		: _dims(dims), _dim(dim), _size(size), _array(array),
		_dim_sliced(slice), _arr_sliced(slice) {}

template <class T>
template <class A>
Tensor <T> &Tensor <T> ::operator=(const Tensor <A> &other)
{
	if (this != &other) {
		_dims = other._dims;
		_size = other._size;

		_dim = new size_t[_dims];
		memcpy(_dim, other._dim, sizeof(size_t) * _dims);

		_array = new T[_size];
		for (size_t i = 0; i < _size; i++)
			_array[i] = static_cast <T> (other._array[i]);
	}

	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
{
	// Faster version for homogenous types (memcpy is faster)
	if (this != &other) {
		_dims = other._dims;
		_size = other._size;

		_dim = new size_t[_dims];
		memcpy(_dim, other._dim, sizeof(size_t) * _dims);

		_array = new T[_size];
		memcpy(_array, other._array, sizeof(T) * _size);
	}

	return *this;
}

/**
 * @brief Deconstructor.
 */
template <class T>
Tensor <T> ::~Tensor()
{
	clear();
}

template <class T>
void Tensor <T> ::clear()
{
	if (!_array && !_dim)
		return;

	if (!_dim_sliced) {

#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
		if (_on_device) {
			if (!_arena)
				throw null_nvarena();

			_arena->free(_dim);
		} else {
			delete[] _dim;
		}
#else
		delete[] _dim;
#endif

	}

	if (!_arr_sliced) {

#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
		if (_on_device) {
			if (!_arena)
				throw null_nvarena();

			_arena->free(_array);
		} else {
			delete[] _array;
		}
#else
		delete[] _array;
#endif

	}

	_array = nullptr;
	_dim = nullptr;
}

/**
 * @brief Returns the size of the tensor.
 *
 * @return the size of the tensor (number of components in the tensor).
 */
template <class T>
__cuda_dual__
size_t Tensor <T> ::size() const
{
	return _size;
}

/*
 * @brief Returns the number of dimensions in the tensor.
 *
 * @return the number of dimensions in the tensor.
 *
template <class T>
__cuda_dual__
size_t Tensor <T> ::dimensions() const
{
	return _dims;
} */

// Return dimensions as a vector
template <class T>
typename Tensor <T> ::shape_type Tensor <T> ::shape() const
{
	shape_type dims(_dims);
	for (size_t i = 0; i < _dims; i++)
		dims[i] = _dim[i];

	return dims;
}

/**
 * @brief Returns the size of a specific dimension. Does not check bounds.
 *
 * @param i the desired index.
 *
 * @return the size of dimension \p i.
 */
template <class T>
__cuda_dual__
size_t Tensor <T> ::dim_size(size_t i) const
{
	return _dim[i];
}

/**
 * @brief Returns the size of a specific dimension. If the index is out of
 * bounds of the number of dimensions, then 1 is returned.
 *
 * @param i the desired index.
 *
 * @return the size of dimension \p i.
 */
template <class T>
__cuda_dual__
size_t Tensor <T> ::safe_dim_size(size_t i) const
{
	return (_dims > i) ? _dim[i] : 1;
}

// Comparison
template <class T>
bool operator==(const Tensor <T> &a, const Tensor <T> &b)
{
	if (a._size != b._size)
		return false;

	for (size_t i = 0; i < a._size; i++) {
		if (a._array[i] != b._array[i])
			return false;
	}

	return true;
}

template <class T>
bool operator!=(const Tensor <T> &a, const Tensor <T> &b)
{
	return !(a == b);
}

// Addition and subtraction
template <class T>
Tensor <T> operator+(const Tensor <T> &a, const Tensor <T> &b)
{
	// TODO: make an exception class
	if (a._size != b._size)		// TODO: this doesnt fully check, make a struct for size info
		throw "Tensor sizes do not match";

	Tensor <T> c(a.shape());

	for (size_t i = 0; i < a._size; i++)
		c._array[i] = a._array[i] + b._array[i];

	return c;
}

// TODO: make += and -= instead
template <class T>
Tensor <T> operator-(const Tensor <T> &a, const Tensor <T> &b)
{
	if (a._size != b._size)
		throw "Tensor size mismatch";

	Tensor <T> c(a.shape());

	for (size_t i = 0; i < a._size; i++)
		c._array[i] = a._array[i] - b._array[i];

	return c;
}

// Indexing
template <class T>
const T &Tensor <T> ::get(size_t i) const
{
	return _array[i];
}

// Applying element-wise transformations
template <class T>
Tensor <T> Tensor <T> ::transform(T (*func)(const T &)) const
{
	Tensor <T> result = *this;

	for (size_t i = 0; i < _size; i++)
		result._array[i] = func(_array[i]);

	return result;
}

template <class T>
Tensor <T> Tensor <T> ::transform(const std::function <T (const T &)> &ftn) const
{
	Tensor <T> out = *this;
	for (int i = 0; i < out._size; i++)
		out._array[i] = ftn(out._array[i]);
	return out;
}

// Element-wise operations
template <class T>
Tensor <T> multiply(const Tensor <T> &a, const Tensor <T> &b)
{
	// TODO: check dimensions (make a helper function for this)
	Tensor <T> c(a.shape());
	for (int i = 0; i < a._size; i++)
		c._array[i] = a._array[i] * b._array[i];
	return c;
}

template <class T>
Tensor <T> divide(const Tensor <T> &a, const Tensor <T> &b)
{
	// TODO: check dimensions (make a helper function for this)
	Tensor <T> c(a.shape());
	for (int i = 0; i < a._size; i++)
		c._array[i] = a._array[i] / b._array[i];
	return c;
}

// Allow comparison of Tensor shapes
template <class A, class B>
bool operator==(const typename Tensor <A> ::shape_type &s1,
		const typename Tensor <B> ::shape_type &s2)
{
	// Ensure same dimension
	if (s1.size() != s2.size())
		return false;

	// Compare each dimension
	for (int i = 0; i < s1.size(); i++) {
		if (s1[i] != s2[i])
			return false;
	}

	return true;
}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim, const std::vector <T> &arr)
		: _dims(dim.size())
{

#ifdef _CUDA_ARCH_

	_on_device = false;

#endif

	_dim = new size_t[_dims];

	size_t prod = 1;
	for (size_t i = 0; i < _dims; i++) {
		prod *= dim[i];

		_dim[i] = dim[i];
	}

	_size = prod;

	if (_size <= 0)
		throw bad_dimensions();

	if (arr.size() != _size)
		throw dimension_mismatch();

	_array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		_array[i] = arr[i];
}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim)
		: _dims(dim.size())
{
	_dim = new size_t[_dims];

	size_t prod = 1;
	for (size_t i = 0; i < _dims; i++) {
		prod *= dim[i];

		_dim[i] = dim[i];
	}

	_size = prod;

	if (!_size)
		return;

	_array = new T[prod];
}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim, const T &def)
		: _dims(dim.size())
{
	_dim = new size_t[_dims];

	size_t prod = 1;
	for (size_t i = 0; i < _dims; i++) {
		prod *= dim[i];

		_dim[i] = dim[i];
	}

	_size = prod;

	if (!_size)
		return;

	_array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		_array[i] = def;
}

template <class T>
bool Tensor <T> ::good() const
{
	return _array != nullptr;
}

// Actions
template <class T>
void Tensor <T> ::nullify(long double p, const utility::Interval <1> &i)
{
	for (size_t k = 0; k < _size; k++) {
		if (p > i.uniform())
			_array[k] = T(0);
	}
}

// Index
template <class T>
T &Tensor <T> ::operator[](const std::vector <size_t> &indices)
{
	size_t full = 0;

	assert(indices.size() == _dims);
	for (size_t i = 0; i < _dims; i++)
		full += indices[i] * _dim[_dims - (i + 1)];

	return _array[full];
}

template <class T>
const T &Tensor <T> ::operator[](const std::vector <size_t> &indices) const
{
	size_t full = 0;

	assert(indices.size() == _dims);
	for (size_t i = 0; i < _dims; i++)
		full += indices[i] * _dim[_dims - (i + 1)];

	return _array[full];
}

// Arithmetic
template <class T>
void Tensor <T> ::operator*=(const T &x)
{
	for (size_t i = 0; i < _size; i++)
		_array[i] *= x;
}

template <class T>
void Tensor <T> ::operator/=(const T &x)
{
	for (size_t i = 0; i < _size; i++)
		_array[i] /= x;
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
	os << print(ts._array, ts._size, ts._dim, 0, ts._dims - 1);

	return os;
}

}

#endif
