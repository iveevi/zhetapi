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
		size_t		offset = 0;
		size_t		elements = 0;
		size_t		dimensions = 0;
		
		// The dimension sizes,
		// partial sizes,
		// slices
		size_t		*array = nullptr;
		size_t		*partials = nullptr;
		slice_type	*slices = nullptr;

		// Constructors
		_shape_info() = default;

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
			offset = other.offset;
			elements = other.elements;
			dimensions = other.dimensions;

			array = new size_t[dimensions];
			partials = new size_t[dimensions];
			slices = new slice_type[dimensions];

			for (size_t i = 0; i < dimensions; i++) {
				array[i] = other.array[i];
				slices[i] = other.slices[i];
				partials[i] = other.partials[i];
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

				offset = other.offset;
				elements = other.elements;
				dimensions = other.dimensions;

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

		// Translate an index using the shape and slices
		size_t translate(size_t index) const {
			size_t result = 0;
			size_t rem = index;

			for (size_t i = 0; i < dimensions; i++) {
				size_t q = rem / partials[i];
				rem = rem % partials[i];

				if (slices[i] == all)
					result += q * partials[i];
				else
					result += slices[i].compute(q) * partials[i];
			}

			return result + offset;
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

		// Slicing a shape info (dimenion remains the same)
		_shape_info slice(const slice_type &slice) const {
			// Skip if slice is all
			if (slice == all)
				return *this;

			// Copy shape info
			_shape_info result = *this;

			// Modify first dimension only
			slice_type nrange = slices[0];
			if (nrange == all)
				nrange = slice;
			else
				nrange = nrange(slice);
			
			result.slices[0] = nrange;
			result.array[0] = nrange.size();
			result.recalculate();

			return result;
		}

		// Indexing a shape info (dimenion - 1)
		_shape_info index(size_t i) const {
			// TODO: edge cases

			// Copy shape info
			_shape_info result;

			result.elements = elements/array[0];
			result.dimensions = dimensions - 1;

			result.array = new size_t[dimensions - 1];
			result.partials = new size_t[dimensions - 1];
			result.slices = new slice_type[dimensions - 1];

			for (size_t i = 0; i < dimensions - 1; i++) {
				result.array[i] = array[i + 1];
				result.partials[i] = partials[i + 1];
				result.slices[i] = slices[i + 1];
			}

			result.offset = offset + i * partials[0];

			return result;
		}

		// Reshape a shape info
		void reshape(const shape_type &shape) {
			// Make sure there are equal number of elements
			size_t elements = 1;
			for (size_t i = 0; i < shape.size(); i++)
				elements *= shape[i];

			if (elements != this->elements) {
				throw std::invalid_argument(
					"Tensor::_shape_info::reshape:"
					" shape does not preserve size"
				);
			}

			// Free previous arrays
			if (array != nullptr)
				delete[] array;

			if (slices != nullptr)
				delete[] slices;

			if (partials != nullptr)
				delete[] partials;

			// Copy new arrays
			dimensions = shape.size();

			array = new size_t[dimensions];
			partials = new size_t[dimensions];
			slices = new slice_type[dimensions];

			for (size_t i = 0; i < dimensions; i++) {
				array[i] = shape[i];
				partials[i] = 1;
				slices[i] = all;
			}
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
public:
	// Essential constructors
	Tensor();

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

	// Initializer list constructor
	Tensor(const std::initializer_list <T> &);

	template <class A>
	Tensor &operator=(const Tensor <A> &);

	// Explicit copy
	Tensor copy() const;

	// Getters and setters
	size_t size() const;		// Get the # of elements
	size_t dimensions() const;
	size_t dimension(size_t) const;
	shape_type shape() const;	// Get shape of tensor

	// Reshape tensor
	void reshape(const shape_type &);

	// Indexing and slicing
	const T &get(size_t) const;	// Get scalar element

	// Slices
	Tensor <T> operator[](size_t) const;
	Tensor <T> operator[](const slice_type &) const;

	// As a vector
	template <class U>
	std::vector <U> as_vector() const;

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

	// Exceptions
	class shape_mismatch : public std::runtime_error {
	public:
		shape_mismatch(const std::string &loc)
				: std::runtime_error(loc +
				": shapes are not matching") {}
	};

	class index_out_of_range : public std::runtime_error {
	public:
		index_out_of_range(size_t i)
				: std::runtime_error("Index " +
				std::to_string(i) + " is out of range") {}
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

template <class T>
Tensor <T> ::Tensor(const std::initializer_list <T> &arr)
		: _shape({arr.size()})
{
	_array.reset(new T[_shape.elements]);
	size_t i = 0;
	for (auto it = arr.begin(); it != arr.end(); it++)
		_array[i++] = *it;
}

///////////////////////
// Essential methods //
///////////////////////

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

// Explicit copy method
template <class T>
Tensor <T> Tensor <T> ::copy() const
{
	return Tensor(
		shape(),
		[&](size_t i) -> T {
			return get(i);
		}
	);
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

// Reshape
template <class T>
void Tensor <T> ::reshape(const shape_type &dim)
{
	_shape.reshape(dim);
}

//////////////////////////
// Indexing and slicing //
//////////////////////////

template <class T>
const T &Tensor <T> ::get(size_t i) const
{
	if (i >= size())
		throw index_out_of_range(i);

	size_t ti = _shape.translate(i);
	return _array[ti];
}

template <class T>
Tensor <T> Tensor <T> ::operator[](size_t i) const
{
	// If only one dimension, return get
	if (dimensions() == 1)
		return get(i);

	Tensor <T> ret;

	// Set properties
	ret._shape = _shape.index(i);
	ret._array = _array;
	return ret;
}

template <class T>
Tensor <T> Tensor <T> ::operator[](const Tensor <T> ::slice_type &slice) const
{
	Tensor <T> ret;

	// Set properties
	ret._shape = _shape.slice(slice);
	ret._array = _array;
	return ret;
}

// As a vector
template <class T>
template <class U>
std::vector <U> Tensor <T> ::as_vector() const
{
	std::vector <U> ret(size());
	for (size_t i = 0; i < size(); i++)
		ret[i] = static_cast <U> (get(i));
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
bool operator==(const Tensor <T> &a, const Tensor <T> &b)
{
	// TODO: static member (SFINAE)
	// TODO: use arithmetic_kernel
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

// Printing method
template <class T>
std::string Tensor <T> ::print() const
{
	// If the tensor is empty, return "[]"
	if (size() == 0)
		return "[]";

	// String to return
	std::string out = "[";

	// If only one dimension
	if (dimensions() == 1) {
		for (int i = 0; i < size(); i++) {
			out += std::to_string(get(i));
			if (i < size() - 1)
				out += ", ";
		}

		return out + "]";
	}

	// Otherwise recurse through each slice
	for (int i = 0; i < dimension(0); i++) {
		out += this->operator[](i).print();
		if (i < dimension(0) - 1)
			out += ", ";
	}

	return out + "]";
}

template <class T>
std::ostream &operator<<(std::ostream &os, const Tensor <T> &ts)
{
	return (os << ts.print());
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
