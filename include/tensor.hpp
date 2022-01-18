#ifndef TENSOR_H_
#define TENSOR_H_

// Standard headers
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Library headers
#include "std/interval.hpp"

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
public:
	// Public type aliases
	using value_type = T;
	using shape_type = std::vector <std::size_t>;

// TODO: should be private, matrix and vector should not be able to access
protected:
	T *	_array		= nullptr;
	bool	_arr_sliced	= false;

	// TODO: also wrap array in a struct

	// Shape informations
	struct _shape_info {
		// Data
		size_t dimensions = 0;
		size_t elements = 0;
		size_t *array = nullptr;

		// TODO: slicing

		// Constructors
		_shape_info() = default;

		// TODO: varaidic constructor
		// TODO: initializer list

		// From shape_type
		_shape_info(const shape_type &shape) {
			dimensions = shape.size();
			elements = 1;

			array = new size_t[dimensions];
			for (size_t i = 0; i < dimensions; i++) {
				array[i] = shape[i];
				elements *= shape[i];
			}

			if (elements < 0) {
				throw std::invalid_argument(
					"Tensor::_shape_info:"
					" shape is invalid"
				);
			}
		}

		// Copy constructor
		_shape_info(const _shape_info &other) {
			dimensions = other.dimensions;
			elements = other.elements;

			array = new size_t[dimensions];
			for (size_t i = 0; i < dimensions; i++)
				array[i] = other.array[i];
		}

		// Assignment operator
		_shape_info &operator=(const _shape_info &other) {
			if (this != &other) {
				if (array != nullptr
					&& dimensions != other.dimensions)
					delete [] array;

				dimensions = other.dimensions;
				elements = other.elements;

				array = new size_t[dimensions];
				for (size_t i = 0; i < dimensions; i++)
					array[i] = other.array[i];
			}

			return *this;
		}

		// Destructor
		~_shape_info() {
			if (array != nullptr)
				delete [] array;
		}

		// Convert to shape_type
		shape_type to_shape_type() const {
			shape_type shape;

			for (size_t i = 0; i < dimensions; i++)
				shape.push_back(array[i]);

			return shape;
		}

		// Boolean comparison
		bool operator==(const _shape_info &other) const {
			if (dimensions != other.dimensions)
				return false;

			for (size_t i = 0; i < dimensions; i++) {
				if (array[i] != other.array[i])
					return false;
			}

			return true;
		}

		bool operator!=(const _shape_info &other) const {
			return !(*this == other);
		}
	} _shape;

	// Private methods
	void _clear();
public:
	// Essential constructors
	Tensor();
	Tensor(const Tensor &);

	template <class A>
	Tensor(const Tensor <A> &);

	// Single element
	Tensor(const T &);

	// TODO: replace with variadic size constructor
	Tensor(size_t, size_t);

	// TODO: make more memory safe
	Tensor(size_t, size_t *, size_t, T *, bool = true);

	// Shape constructors
	explicit Tensor(const shape_type &);
	Tensor(const shape_type &, const T &);
	Tensor(const shape_type &, const std::vector <T> &);

	// Destructor
	virtual ~Tensor();

	// Essential methods
	Tensor &operator=(const Tensor &);

	template <class A>
	Tensor &operator=(const Tensor <A> &);

	// Getters and setters
	size_t size() const;		// Get the # of elements
	size_t dimensions() const;
	size_t dimension(size_t) const;
	shape_type shape() const;	// Get shape of tensor
	const T &get(size_t) const;	// Get scalar element

	// TODO: iterators

	// Actions
	void nullify(long double, const utility::Interval <1> &);

	// Apply a function to each element of the tensor
	Tensor <T> transform(T (*)(const T &)) const;
	Tensor <T> transform(const std::function <T (const T &)> &) const;

	// Properties
	// TODO: is this needed?
	bool good() const; // operator bool() const; ?

	// Arithmetic modifiers
	Tensor <T> &operator*=(const T &);
	Tensor <T> &operator/=(const T &);

	Tensor <T> &operator+=(const Tensor <T> &);
	Tensor <T> &operator-=(const Tensor <T> &);

	// Arithmetic
	template <class U>
	friend Tensor <U> operator+(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend Tensor <U> operator-(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend Tensor <U> multiply(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend Tensor <U> divide(const Tensor <U> &, const Tensor <U> &);

	// Boolean operators
	// TODO: specialize for floating types to account for error
	template <class U>
	friend bool operator==(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend bool operator!=(const Tensor <U> &, const Tensor <U> &);

	// Printing functions
	std::string print() const;

	template <class U>
	friend std::ostream &operator<<(std::ostream &, const Tensor <U> &);

	// Shape mismatch exception
	class shape_mismatch : public std::runtime_error {
	public:
		shape_mismatch(const std::string &loc)
				: std::runtime_error(loc +
				": shapes are not matching") {}
	};
};

/////////////////////////
// Tensor constructors //
/////////////////////////

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
		: _shape(other._shape)
{
	_array = new T[_shape.elements];
	memcpy(_array, other._array, sizeof(T) * _shape.elements);
}

/**
 * @brief Heterogenous (with respect to the component type) copy constructor.
 *
 * @param other the reference vector (to be copied from).
 */
template <class T>
template <class A>
Tensor <T> ::Tensor(const Tensor <A> &other)
		: _shape(other._shape)
{
	_array = new T[_shape.elements];
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = static_cast <T> (other._array[i]);
}

// Single element
// TODO: should delegate to another constructor
template <class T>
Tensor <T> ::Tensor(const T &value)
		: _shape({1})
{
	_array = new T[1];
	_array[0] = value;
}

template <class T>
Tensor <T> ::Tensor(size_t rows, size_t cols)
		: _shape({rows, cols})
{
	_array = new T[_shape.elements];
}

/*
template <class T>
Tensor <T> ::Tensor(size_t dims, size_t *dim, size_t size, T *array, bool slice)
		: _dims(dims), _dim(dim), _size(size), _array(array),
		_dim_sliced(slice), _arr_sliced(slice) {} */


template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const std::vector <T> &arr)
		: _shape(dim)
{
	if (arr.size() != _shape.elements)
		throw shape_mismatch(__PRETTY_FUNCTION__);

	_array = new T[_shape.elements];
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = arr[i];
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim)
		: _shape(dim)
{
	_array = new T[_shape.elements];
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const T &def)
		: _shape(dim)
{
	_array = new T[_shape.elements];
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = def;
}

// TODO: remove once _array in a struct
/**
 * @brief Deconstructor.
 */
template <class T>
Tensor <T> ::~Tensor()
{
	_clear();
}

// TODO: remove once _array in a struct
template <class T>
void Tensor <T> ::_clear()
{
	if (_array && !_arr_sliced)
		delete[] _array;
}

///////////////////////
// Essential methods //
///////////////////////

template <class T>
Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
{
	// Faster version for homogenous types (memcpy is faster)
	if (this != &other) {
		_shape = other._shape;

		_array = new T[_shape.elements];
		memcpy(_array, other._array, sizeof(T) * _shape.elements);
	}

	return *this;
}

template <class T>
template <class A>
Tensor <T> &Tensor <T> ::operator=(const Tensor <A> &other)
{
	if (this != &other) {
		_shape = other._shape;

		_array = new T[_shape.elements];
		for (size_t i = 0; i < _shape.elements; i++)
			_array[i] = static_cast <T> (other._array[i]);
	}

	return *this;
}

/////////////////////////
// Getters and setters //
/////////////////////////

/**
 * @brief Returns the size of the tensor.
 *
 * @return the size of the tensor (number of components in the tensor).
 */
template <class T>
size_t Tensor <T> ::size() const
{
	return _shape.elements;
}

template <class T>
size_t Tensor <T> ::dimensions() const
{
	return _shape.dimensions;
}

template <class T>
size_t Tensor <T> ::dimension(size_t index) const
{
	return _shape.array[index];
}

// Return dimensions as a vector
template <class T>
typename Tensor <T> ::shape_type Tensor <T> ::shape() const
{
	return _shape.to_shape_type();
}

// Indexing
template <class T>
const T &Tensor <T> ::get(size_t i) const
{
	return _array[i];
}

////////////////
// Properties //
////////////////

template <class T>
bool Tensor <T> ::good() const
{
	return _array != nullptr;
}

/////////////
// Actions //
/////////////

// TODO: should not be a method
template <class T>
void Tensor <T> ::nullify(long double p, const utility::Interval <1> &i)
{
	for (size_t k = 0; k < size(); k++) {
		if (p > i.uniform())
			_array[k] = T(0);
	}
}

// Applying element-wise transformations
template <class T>
Tensor <T> Tensor <T> ::transform(T (*func)(const T &)) const
{
	Tensor <T> result = *this;

	for (size_t i = 0; i < result.size(); i++)
		result._array[i] = func(_array[i]);

	return result;
}

template <class T>
Tensor <T> Tensor <T> ::transform(const std::function <T (const T &)> &ftn) const
{
	Tensor <T> out = *this;
	for (int i = 0; i < out.size(); i++)
		out._array[i] = ftn(out._array[i]);
	return out;
}

//////////////////////////
// Arithmetic modifiers //
//////////////////////////

template <class T>
Tensor <T> &Tensor <T> ::operator*=(const T &x)
{
	for (size_t i = 0; i < size(); i++)
		_array[i] *= x;
	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator/=(const T &x)
{
	for (size_t i = 0; i < size(); i++)
		_array[i] /= x;
	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator+=(const Tensor <T> &ts)
{
	if (_shape != ts._shape)
		throw shape_mismatch(__PRETTY_FUNCTION__);

	for (size_t i = 0; i < size(); i++)
		_array[i] += ts._array[i];

	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator-=(const Tensor <T> &ts)
{
	if (shape() != ts.shape())
		throw shape_mismatch(__PRETTY_FUNCTION__);

	for (size_t i = 0; i < size(); i++)
		_array[i] -= ts._array[i];

	return *this;
}

///////////////////////////
// Arithmetic operations //
///////////////////////////

template <class T>
Tensor <T> operator+(const Tensor <T> &a, const Tensor <T> &b)
{
	Tensor <T> c(a);
	return (c += b);
}

template <class T>
Tensor <T> operator-(const Tensor <T> &a, const Tensor <T> &b)
{
	Tensor <T> c(a);
	return (c -= b);
}

template <class T>
Tensor <T> multiply(const Tensor <T> &a, const Tensor <T> &b)
{
	if (a.shape() != b.shape())
		throw typename Tensor <T> ::shape_mismatch(__PRETTY_FUNCTION__);

	Tensor <T> c(a.shape());
	for (int i = 0; i < a.size(); i++)
		c._array[i] = a._array[i] * b._array[i];
	return c;
}

template <class T>
Tensor <T> divide(const Tensor <T> &a, const Tensor <T> &b)
{
	if (a.shape() != b.shape())
		throw typename Tensor <T> ::shape_mismatch(__PRETTY_FUNCTION__);

	Tensor <T> c(a.shape());
	for (int i = 0; i < a.size(); i++)
		c._array[i] = a._array[i] / b._array[i];
	return c;
}

// Boolean operators
template <class T>
// TODO: SFINAE overload for floating types
bool operator==(const Tensor <T> &a, const Tensor <T> &b)
{
	// TODO: static member (SFINAE)
	static const T epsilon = 1e-5;

	if (a._shape != b._shape)
		return false;

	for (size_t i = 0; i < a.size(); i++) {
		if (std::is_floating_point <T> ::value) {
			if (std::abs(a._array[i] - b._array[i]) > epsilon)
				return false;
		} else {
			if (a._array[i] != b._array[i])
				return false;
		}
	}

	return true;
}

template <class T>
bool operator!=(const Tensor <T> &a, const Tensor <T> &b)
{
	return !(a == b);
}

////////////////////////
// Printing functions //
////////////////////////

// TODO: turn into a method
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
	// TODO: use the method
	os << print(ts._array, ts.size(), ts._shape.array, 0, ts.dimensions() - 1);

	return os;
}

/////////////////////
// Extra functions //
/////////////////////

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

}

#endif
