#ifndef TENSOR_H_
#define TENSOR_H_

// Standard headers
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Library headers
#include "range.hpp"
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
	using slice_type = Range <int>;

// TODO: should be private, matrix and vector should not be able to access
protected:
	// T *	_array		= nullptr;
	std::shared_ptr <T []> _array;

	// TODO: also wrap array in a struct

	// Shape informations
	struct _shape_info {
		// Data
		size_t elements = 0;
		size_t dimensions = 0;
		
		// The dimension sizes,
		// partial sizes,
		// slices
		size_t *array = nullptr;
		size_t *partials = nullptr;
		slice_type *slices = nullptr;

		// TODO: slicing

		// Constructors
		_shape_info() = default;

		// TODO: varaidic constructor
		// TODO: initializer list

		// From shape_type
		_shape_info(const shape_type &shape)
				: elements(1),
				dimensions(shape.size()) {
			array = new size_t[dimensions];
			partials = new size_t[dimensions];
			slices = new slice_type[dimensions];

			for (int i = dimensions - 1; i >= 0; i--) {
				array[i] = shape[i];
				slices[i] = all;
				partials[i] = elements;
				elements *= shape[i];
			}

			// TODO: this check is redundant...
			// at least make elements an integer
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
			partials = new size_t[dimensions];
			slices = new slice_type[dimensions];

			for (size_t i = 0; i < dimensions; i++) {
				array[i] = other.array[i];
				slices[i] = other.slices[i];
			}
		}

		// Assignment operator
		_shape_info &operator=(const _shape_info &other) {
			if (this != &other) {
				if (array != nullptr
					&& dimensions != other.dimensions) {
					delete [] array;
					delete [] partials;
					delete [] slices;
				}

				dimensions = other.dimensions;
				elements = other.elements;

				array = new size_t[dimensions];
				partials = new size_t[dimensions];
				slices = new slice_type[dimensions];

				for (size_t i = 0; i < dimensions; i++) {
					array[i] = other.array[i];
					partials[i] = other.partials[i];
					slices[i] = other.slices[i];
				}
			}

			return *this;
		}

		// Destructor
		~_shape_info() {
			if (array != nullptr)
				delete[] array;

			if (slices != nullptr)
				delete[] slices;

			if (partials != nullptr)
				delete[] partials;
		}

		// Recalculate properties based on slices
		void recalculate() {
			elements = 1;

			for (int i = dimensions - 1; i >= 0; i--) {
				partials[i] = elements;
				if (slices[i] == all)
					elements *= array[i];
				else
					elements *= slices[i].size();
			}
		}

		// Slicing a shape info
		_shape_info slice(const slice_type &slice, int dim) const {
			// Skip if slice is all
			if (slice == all)
				return *this;

			// Copy shape info
			_shape_info new_shape(*this);
			new_shape.array[dim] = slice.size();
			new_shape.slices[dim] = slice;
			new_shape.recalculate();

			return new_shape;
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
	// void _clear();
public:
	// Essential constructors
	Tensor();
	Tensor(const Tensor &);		// TODO: this isnt needed with shared_ptr

	template <class A>
	Tensor(const Tensor <A> &);

	// Single element
	Tensor(const T &);

	// TODO: replace with variadic size constructor
	Tensor(size_t, size_t);

	// Shape constructors
	explicit Tensor(const shape_type &);
	Tensor(const shape_type &, const T &);
	Tensor(const shape_type &, const std::vector <T> &);
	Tensor(const shape_type &, const std::initializer_list <T> &);
	Tensor(const shape_type &, const std::function <T (size_t)> &);

	// Destructor
	// virtual ~Tensor();

	// Essential methods
	Tensor &operator=(const Tensor &); // TODO: this isnt needed with shared_ptr

	template <class A>
	Tensor &operator=(const Tensor <A> &);

	// Getters and setters
	size_t size() const;		// Get the # of elements
	size_t dimensions() const;
	size_t dimension(size_t) const;
	shape_type shape() const;	// Get shape of tensor

	// Indexing and slicing
	const T &get(size_t) const;	// Get scalar element

	Tensor <T> operator[](const slice_type &) const;

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
		: _array(other._array), _shape(other._shape) {}

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
	_array.reset(new T[_shape.elements]);
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = static_cast <T> (other._array[i]);
}

// Single element
// TODO: should delegate to another constructor
template <class T>
Tensor <T> ::Tensor(const T &value)
		: _shape({1})
{
	_array.reset(new T[1]);
	_array[0] = value;
}

template <class T>
Tensor <T> ::Tensor(size_t rows, size_t cols)
		: _shape({rows, cols})
{
	_array.reset(new T[_shape.elements]);
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim)
		: _shape(dim)
{
	_array.reset(new T[_shape.elements]);
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const T &def)
		: _shape(dim)
{
	_array.reset(new T[_shape.elements]);
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = def;
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const std::vector <T> &arr)
		: _shape(dim)
{
	if (arr.size() != _shape.elements)
		throw shape_mismatch(__PRETTY_FUNCTION__);

	_array.reset(new T[_shape.elements]);
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = arr[i];
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const std::initializer_list <T> &arr)
		: _shape(dim)
{
	if (arr.size() != _shape.elements)
		throw shape_mismatch(__PRETTY_FUNCTION__);

	_array.reset(new T[_shape.elements]);
	size_t i = 0;
	for (auto it = arr.begin(); it != arr.end(); it++)
		_array[i++] = *it;
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const std::function <T (size_t)> &f)
		: _shape(dim)
{
	_array.reset(new T[_shape.elements]);
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = f(i);
}

/*

// TODO: remove once _array in a struct
template <class T>
Tensor <T> ::~Tensor()
{
	_clear();
}

// TODO: remove once _array in a struct
template <class T>
void Tensor <T> ::_clear()
{
	if (_array)
		delete[] _array;
} */

///////////////////////
// Essential methods //
///////////////////////

template <class T>
Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
{
	// Faster version for homogenous types (memcpy is faster)
	if (this != &other) {
		_shape = other._shape;
		_array = other._array;

		// _array = new T[_shape.elements];
		// memcpy(_array, other._array, sizeof(T) * _shape.elements);
	}

	return *this;
}

template <class T>
template <class A>
Tensor <T> &Tensor <T> ::operator=(const Tensor <A> &other)
{
	if (this != &other) {
		_shape = other._shape;

		_array.reset(new T[_shape.elements]);
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

//////////////////////////
// Indexing and slicing //
//////////////////////////

template <class T>
const T &Tensor <T> ::get(size_t i) const
{
	return _array[i];
}

template <class T>
Tensor <T> Tensor <T> ::operator[](const Tensor <T> ::slice_type &slice) const
{
	// TODO: redirect to helper method
	Tensor <T> ret;

	// Set properties
	ret._shape = _shape.slice(slice, 0);
	ret._array = _array;
	return ret;
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

// Print helper
template <class T>
std::string print_tensor(T *array, size_t size,
		size_t *dimension_array,
		typename Tensor <T> ::slice_type *slice_array,
		size_t dim,
		size_t max_dim)
{
	if (size == 0)
		return "[]";

	std::string out = "[";

	// Size of each dimension
	size_t dim_size = size / dimension_array[dim];

	// Get the range
	typename Tensor <T> ::slice_type range = slice_array[dim];

	// Change the range if it is all
	if (range == all) {
		range = typename Tensor <T> ::slice_type(
			0, dimension_array[dim]
		);
	}

	auto itr = range.begin();
	auto end = range.end();

	while (itr < end) {
		T *current = array + (*itr) * dim_size;

		if (dim == max_dim) {
			out += std::to_string(*current);
		} else {
			out += print_tensor(
				current, dim_size,
				dimension_array,
				slice_array,
				dim + 1, max_dim
			);
		}

		itr++;
		if (itr < end)
			out += ", ";
	}

	return out + "]";
}


// Printing method
template <class T>
std::string Tensor <T> ::print() const
{
	return print_tensor(
		_array.get(), _shape.elements,
		_shape.array, _shape.slices,
		0, _shape.dimensions - 1
	);
}

template <class T>
std::ostream &operator<<(std::ostream &os, const Tensor <T> &ts)
{
	// TODO: use the method
	os << ts.print();

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
