#ifndef ZHETAPI_TENSOR_H_
#define ZHETAPI_TENSOR_H_

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

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
#include "allocator.hpp"
#include "range.hpp"
#include "std/interval.hpp"
#include "field.hpp"

namespace zhetapi {

namespace utility {

// TODO: move elsewhere
template <size_t N>
class Interval;

}

// Shape information for tensors
// TODO: place into dedicated header
struct _shape_info {
	using shape_type = std::vector <size_t>;
	using slice_type = Range <int>;

	// Data
	size_t		offset = 0;
	size_t		elements = 0;
	size_t		dimensions = 0;

	// The dimension sizes, partial sizes, slices
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

		for (int i = static_cast <int> (dimensions) - 1; i >= 0; i--) {
			array[i] = shape[i];
			slices[i] = all;
			partials[i] = elements;
			elements *= shape[i];
		}

		/* TODO: this check is redundant...
		if (elements < 0) {
			throw std::invalid_argument(
				"Tensor::_shape_info:"
				" shape is invalid"
			);
		} */
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

	// Index dimension
	size_t operator[](size_t i) const {
		return array[i];
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

			// TODO: this isnt right, needs to consider all
			// previous slices
			slices[i] = all;
		}
	}

	// Flatten a shape info (one dimension)
	void flatten() {
		// Free previous arrays
		if (array != nullptr)
			delete[] array;

		if (slices != nullptr)
			delete[] slices;

		if (partials != nullptr)
			delete[] partials;

		// New shape info
		dimensions = 1;

		array = new size_t[dimensions];
		partials = new size_t[dimensions];
		slices = new slice_type[dimensions];

		array[0] = elements;
		partials[0] = 1;

		// TODO: need to compose all teh slices
		slices[0] = all;
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

	// Check if scalar
	bool is_scalar() const {
		return (dimensions == 1) && (elements == 1);
	}
};

// Tensor class
template <class T>
class Tensor : public Field <T, Tensor <T>> {
public:
	// Public type aliases
	using shape_type = std::vector <std::size_t>;
	using slice_type = Range <int>;
	using value_type = T;
protected:
	// TODO: should be private, matrix and vector should not be able to access
	std::shared_ptr <T []>	_array;
	_shape_info		_shape;

	// Private constructor
	Tensor(const std::shared_ptr <T []> &, const _shape_info &, Variant);
public:
	// Essential constructors
	Tensor();

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
	Tensor(const std::initializer_list <Tensor <T>> &);

	// Explicit copy operations
	Tensor copy() const;
	void copy(const Tensor &); // Copy at most the number of elements

	// Getters and setters
	size_t size() const;		// Get the # of elements
	size_t dimensions() const;
	size_t dimension(size_t) const;
	shape_type shape() const;	// Get shape of tensor

	// Access raw array
	T *data();
	const T *data() const;

	// Reshape tensor
	void reshape(const shape_type &);
	void flatten();

	// Indexing and slicing
	const T &get(size_t) const;	// Get scalar element

	// Slices
	Tensor <T> operator[](size_t) const;
	Tensor <T> operator[](const slice_type &) const;

	// As a vector
	template <class U>
	std::vector <U> as_vector() const;

	// Length (as a vector)
	T length() const;

	// TODO: iterators

	// Actions
	// TODO: put method somewher else...
	void nullify(long double, const utility::Interval <1> &);

	// Apply a function to each element of the tensor
	Tensor <T> transform(T (*)(const T &)) const;
	Tensor <T> transform(const std::function <T (const T &)> &) const;

	Tensor <T> flat() const;

	// Properties
	// TODO: is this needed?
	bool good() const; // operator bool() const; ?
	bool is_scalar() const;

	// Arithmetic modifiers
	Tensor <T> &operator+=(const T &);
	Tensor <T> &operator-=(const T &);

	Tensor <T> &operator+=(const Tensor <T> &);
	Tensor <T> &operator-=(const Tensor <T> &);

	Tensor <T> &operator*=(const T &);
	Tensor <T> &operator/=(const T &);

	Tensor <T> &operator*=(const Tensor &);
	Tensor <T> &operator/=(const Tensor &);

	// Static generators
	static Tensor <T> zeros(const shape_type &);
	static Tensor <T> ones(const shape_type &);

	/* template <class ... Args>
	static Tensor <T> zeros(Args...);

	template <class ... Args>
	static Tensor <T> ones(Args...); */

	// Arithmetic
	template <class U>
	friend Tensor <U> operator-(const U &, const Tensor <U> &);

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

	// Additional functions
	template <class U>
	friend U max(const Tensor <U> &);

	template <class U>
	friend U min(const Tensor <U> &);

	template <class U>
	friend int argmax(const Tensor <U> &);

	template <class U>
	friend int argmin(const Tensor <U> &);

	template <class U>
	friend U sum(const Tensor <U> &);

	template <class U, class F>
	friend U sum(const Tensor <U> &, const F &);

	template <class U>
	friend U dot(const Tensor <U> &, const Tensor <U> &);

	// Printing functions
	std::string str() const;
	std::string verbose() const;

	template <class U>
	friend std::ostream &operator<<(std::ostream &, const Tensor <U> &);

	// Friend other tensors
	template <class U>
	friend class Tensor;

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

	// Configuring operative variants
	static void set_variant(Variant variant) {
		s_variant = variant;
	}
private:
	// Static and member-wise variants
	static Variant s_variant;
	Variant m_variant;
};

// Variant starts as CPU by default
template <class T>
Variant Tensor <T> ::s_variant = eCPU;

/////////////////////////
// Tensor constructors //
/////////////////////////

/**
 * @brief Default constructor.
 */
template <class T>
Tensor <T> ::Tensor() : m_variant(s_variant) {}

// Single element
// TODO: should delegate to another constructor
template <class T>
Tensor <T> ::Tensor(const T &value)
		: _shape({1}), m_variant(s_variant)
{
	_array = detail::make_shared_array <T> (1, m_variant);
	_array[0] = value;
}

template <class T>
Tensor <T> ::Tensor(size_t rows, size_t cols)
		: _shape({rows, cols}), m_variant(s_variant)
{
	_array = detail::make_shared_array <T> (_shape.elements, m_variant);
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim)
		: _shape(dim), m_variant(s_variant)
{
	_array = detail::make_shared_array <T> (_shape.elements, m_variant);
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const T &def)
		: _shape(dim), m_variant(s_variant)
{
	_array = detail::make_shared_array <T> (_shape.elements, m_variant);
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = def;
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const std::vector <T> &arr)
		: _shape(dim), m_variant(s_variant)
{
	if (arr.size() != _shape.elements)
		throw shape_mismatch(__PRETTY_FUNCTION__);

	_array = detail::make_shared_array <T> (_shape.elements, m_variant);
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = arr[i];
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const std::initializer_list <T> &arr)
		: _shape(dim), m_variant(s_variant)
{
	if (arr.size() != _shape.elements)
		throw shape_mismatch(__PRETTY_FUNCTION__);

	_array = detail::make_shared_array <T> (_shape.elements, m_variant);

	size_t i = 0;
	for (auto it = arr.begin(); it != arr.end(); it++)
		_array[i++] = *it;
}

template <class T>
Tensor <T> ::Tensor(const shape_type &dim, const std::function <T (size_t)> &f)
		: _shape(dim), m_variant(s_variant)
{
	_array = detail::make_shared_array <T> (_shape.elements, m_variant);
	for (size_t i = 0; i < _shape.elements; i++)
		_array[i] = f(i);
}

// Initializer list constructors
template <class T>
Tensor <T> ::Tensor(const std::initializer_list <T> &arr)
		: _shape({arr.size()}), m_variant(s_variant)
{
	_array = detail::make_shared_array <T> (_shape.elements, m_variant);

	size_t i = 0;
	for (auto it = arr.begin(); it != arr.end(); it++)
		_array[i++] = *it;
}

template <class T>
Tensor <T> ::Tensor(const std::initializer_list <Tensor <T>> &tensors)
		: m_variant(s_variant)
{
	// Make sure all tensors have the same shape
	_shape_info shape = tensors.begin()->_shape;
	for (auto it = tensors.begin(); it != tensors.end(); it++) {
		if (it->_shape != shape)
			throw shape_mismatch(__PRETTY_FUNCTION__);
	}

	// Create a new dimension
	shape_type dims = shape.to_shape_type();
	dims.insert(dims.begin(), tensors.size());

	_shape = shape_type(dims);
	_array = detail::make_shared_array <T> (_shape.elements, m_variant);

	// Copy the data
	// TODO: detail::copy based on the variant
	size_t i = 0;
	for (auto it = tensors.begin(); it != tensors.end(); it++) {
		// std::copy will be faster
		std::copy(
			it->_array.get(),
			it->_array.get() + shape.elements,
			_array.get() + i
		);

		i += shape.elements;
	}
}

// Private constructor
template <class T>
Tensor <T> ::Tensor(const std::shared_ptr <T []> &array,
		const _shape_info &shape, Variant variant)
		: _shape(shape), _array(array), m_variant(variant) {}

///////////////////////
// Essential methods //
///////////////////////

// Explicit copy methods
template <class T>
Tensor <T> Tensor <T> ::copy() const
{
	auto shape = _shape;
	auto array = detail::make_shared_array <T> (shape.elements, m_variant);
	detail::copy(array, _array, shape.elements, m_variant);
	return Tensor(array, shape, m_variant);
}

template <class T>
void Tensor <T> ::copy(const Tensor &tensor)
{
	// TODO: warning log if fewer elements in the tensor
	// TODO: use copy helper...
#ifdef __CUDACC__
	if (m_variant == eCUDA) {
		if (tensor.m_variant == eCUDA) {
			cudaMemcpy(
				_array.get(), tensor._array.get(),
				_shape.elements * sizeof(T),
				cudaMemcpyDeviceToDevice
			);
		} else {
			cudaMemcpy(
				_array.get(), tensor._array.get(),
				_shape.elements * sizeof(T),
				cudaMemcpyHostToDevice
			);
		}
	} else {
		if (tensor.m_variant == eCUDA) {
			cudaMemcpy(
				_array.get(), tensor._array.get(),
				_shape.elements * sizeof(T),
				cudaMemcpyDeviceToHost
			);
		} else {
			std::copy(
				tensor._array.get(),
				tensor._array.get() + _shape.elements,
				_array.get()
			);
		}
	}
#else
	if (tensor.m_variant != m_variant)
		throw std::runtime_error("Cannot copy between different variants");
	else if (tensor.m_variant == Variant::eCUDA)
		throw std::runtime_error("CUDA variant not enabled");

	std::copy(
		tensor._array.get(),
		tensor._array.get() + _shape.elements,
		_array.get()
	);
#endif
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

// Raw pointer to the data
template <class T>
T *Tensor <T> ::data()
{
	return _array.get();
}

template <class T>
const T *Tensor <T> ::data() const
{
	return _array.get();
}

// Reshape
template <class T>
void Tensor <T> ::reshape(const shape_type &dim)
{
	_shape.reshape(dim);
}

template <class T>
void Tensor <T> ::flatten()
{
	_shape.flatten();
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

template <class T>
bool Tensor <T> ::is_scalar() const
{
	return _shape.is_scalar();
}

template <class T>
T Tensor <T> ::length() const
{
	T sum = 0;
	for (size_t i = 0; i < size(); i++)
		sum += get(i) * get(i);
	return std::sqrt(sum);
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
	Tensor <T> result = copy();
	for (size_t i = 0; i < result.size(); i++)
		result._array[i] = func(_array[i]);
	return result;
}

template <class T>
Tensor <T> Tensor <T> ::transform(const std::function <T (const T &)> &ftn) const
{
	Tensor <T> out = copy();
	for (int i = 0; i < out.size(); i++)
		out._array[i] = ftn(out._array[i]);
	return out;
}

template <class T>
Tensor <T> Tensor <T> ::flat() const
{
	auto ret = copy();
	ret.flatten();
	return ret;
}

///////////////////////
// Static generators //
///////////////////////

template <class T>
Tensor <T> Tensor <T> ::zeros(const shape_type &dim)
{
	return Tensor(dim, T(0));
}

template <class T>
Tensor <T> Tensor <T> ::ones(const shape_type &dim)
{
	return Tensor(dim, T(1));
}

//////////////////////////
// Arithmetic modifiers //
//////////////////////////

template <class T>
Tensor <T> &Tensor <T> ::operator+=(const T &x)
{
#pragma omp parallel for
	for (size_t i = 0; i < size(); i++)
		_array[i] += x;
	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator-=(const T &x)
{
#pragma omp parallel for
	for (size_t i = 0; i < size(); i++)
		_array[i] -= x;
	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator+=(const Tensor <T> &ts)
{
	if (_shape != ts._shape)
		throw shape_mismatch(__PRETTY_FUNCTION__);

// TODO: avoid nested parallelization...
#pragma omp parallel for
	for (long int i = 0; i < size(); i++)
		_array[i] += ts._array[i];

	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator-=(const Tensor <T> &ts)
{
	if (shape() != ts.shape())
		throw shape_mismatch(__PRETTY_FUNCTION__);

#pragma omp parallel for
	for (long int i = 0; i < size(); i++)
		_array[i] -= ts._array[i];

	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator*=(const T &x)
{
#pragma omp parallel for
	for (long int i = 0; i < size(); i++)
		_array[i] *= x;
	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator/=(const T &x)
{
#pragma omp parallel for
	for (long int i = 0; i < size(); i++)
		_array[i] /= x;
	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator*=(const Tensor <T> &other)
{
	// Either scalar multiplication or element-wise multiplication
	if (shape() == other.shape()) {
		// Element-wise multiplication
#pragma omp parallel for
		for (long int i = 0; i < size(); i++)
			_array[i] *= other._array[i];
	} else if (is_scalar()) {
		*this = other * get(0);
	} else if (other.is_scalar()) {
		*this *= other.get(0);
	} else {
		throw typename Tensor <T> ::shape_mismatch(__PRETTY_FUNCTION__);
	}

	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator/=(const Tensor <T> &other)
{
	// Either scalar division or element-wise division
	if (shape() == other.shape()) {
		// Element-wise division
#pragma omp parallel for
		for (long int i = 0; i < size(); i++)
			_array[i] /= other._array[i];
	} else if (is_scalar()) {
		*this = other / get(0);
	} else if (other.is_scalar()) {
		*this /= other.get(0);
	} else {
		throw typename Tensor <T> ::shape_mismatch(__PRETTY_FUNCTION__);
	}

	return *this;
}

///////////////////////////
// Arithmetic operations //
///////////////////////////

template <class T>
Tensor <T> operator+(const Tensor <T> &a, const Tensor <T> &b)
{
	Tensor <T> c = a.copy();
	return (c += b);
}

template <class T>
Tensor <T> operator+(const Tensor <T> &a, const T &b)
{
	Tensor <T> c = a.copy();
	return (c += b);
}

template <class T>
Tensor <T> operator+(const T &a, const Tensor <T> &b)
{
	Tensor <T> c = b.copy();
	return (c += a);
}

template <class T>
Tensor <T> operator-(const Tensor <T> &a, const Tensor <T> &b)
{
	Tensor <T> c = a.copy();
	return (c -= b);
}

template <class T>
Tensor <T> operator-(const Tensor <T> &a, const T &b)
{
	Tensor <T> c = a.copy();
	return (c -= b);
}

template <class T>
Tensor <T> operator-(const T &a, const Tensor <T> &b)
{
	Tensor <T> c = b.copy();
	for (size_t i = 0; i < c.size(); i++)
		c._array[i] = a - c._array[i];
	return c;
}

// TODO: these are redundant now
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

template <class T>
Tensor <T> operator*(const Tensor <T> &a, const Tensor <T> &b)
{
	Tensor <T> c = a.copy();
	return (c *= b);
}

template <class T>
Tensor <T> operator*(const Tensor <T> &a, const T &b)
{
	Tensor <T> c = a.copy();
	return (c *= b);
}

template <class T>
Tensor <T> operator*(const T &a, const Tensor <T> &b)
{
	Tensor <T> c = b.copy();
	return (c *= a);
}

template <class T>
Tensor <T> operator/(const Tensor <T> &a, const Tensor <T> &b)
{
	Tensor <T> c = a.copy();
	return (c /= b);
}

template <class T>
Tensor <T> operator/(const Tensor <T> &a, const T &b)
{
	Tensor <T> c = a.copy();
	return (c /= b);
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

// Additional functions
template <class T>
T max(const Tensor <T> &a)
{
	T max = a.get(0);
	for (int i = 1; i < a.size(); i++)
		if (a._array[i] > max)
			max = a._array[i];
	return max;
}

template <class T>
T min(const Tensor <T> &a)
{
	T min = a.get(0);
	for (int i = 1; i < a.size(); i++)
		if (a._array[i] < min)
			min = a._array[i];
	return min;
}

template <class T>
int argmax(const Tensor <T> &a)
{
	T max = a.get(0);

	int argmax = 0;
	for (int i = 1; i < a.size(); i++) {
		if (a._array[i] > max) {
			max = a._array[i];
			argmax = i;
		}
	}

	return argmax;
}

template <class T>
int argmin(const Tensor <T> &a)
{
	T min = a.get(0);

	int argmin = 0;
	for (int i = 1; i < a.size(); i++) {
		if (a._array[i] < min) {
			min = a._array[i];
			argmin = i;
		}
	}

	return argmin;
}

template <class T>
T sum(const Tensor <T> &a)
{
	T sum = 0;
	for (int i = 0; i < a.size(); i++)
		sum += a._array[i];
	return sum;
}

template <class T, class F>
T sum(const Tensor <T> &a, const F &f)
{
	T sum = 0;
	for (int i = 0; i < a.size(); i++)
		sum += f(a._array[i]);
	return sum;
}

template <class T>
T dot(const Tensor <T> &a, const Tensor <T> &b)
{
	if (a.shape() != b.shape())
		throw typename Tensor <T> ::shape_mismatch(__PRETTY_FUNCTION__);

	T dot = 0;
	for (int i = 0; i < a.size(); i++)
		dot += a._array[i] * b._array[i];
	return dot;
}

////////////////////////
// Printing functions //
////////////////////////

// Printing method
template <class T>
std::string Tensor <T> ::str() const
{
	// Print shape, then address and variant
	std::string out = "(";

	shape_type shape_list = _shape.to_shape_type();
	for (int i = 0; i < shape_list.size(); i++) {
		out += std::to_string(shape_list[i]);
		if (i < shape_list.size() - 1)
			out += ", ";
		else
			out += ")";
	}

	// TODO: verbose mode
	// std::string variant_str = (m_variant == eCPU) ? "CPU" : "CUDA";
	// out += ") " + std::to_string((long long) this) + " " + variant_str;

	return out;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const Tensor <T> &ts)
{
	return (os << ts.str());
}

// Verbose printing method (all elements)
template <class T>
std::string verbose_str(T *data, const _shape_info &shape)
{
	// If the tensor is empty, return "[]"
	size_t size = shape.elements;
	if (size == 0)
		return "[]";

	// String to return
	std::string out = "[";

	// If only one dimension
	if (shape.dimensions == 1) {
		for (int i = 0; i < size; i++) {
			int index = shape.translate(i);
			out += std::to_string(data[index]);
			if (i < size - 1)
				out += ", ";
		}

		return out + "]";
	}

	// Otherwise recurse through each slice
	size_t stride = size/shape[0];
	_shape_info sub = shape.index(0);

	for (int i = 0; i < shape[0]; i++) {
		out += verbose_str(data + i * stride, sub);
		if (i < shape[0] - 1)
			out += ", ";
	}

	return out + "]";
}

template <class T>
std::string Tensor <T> ::verbose() const
{
	if (m_variant == eCPU) {
		return verbose_str(_array.get(), _shape);
	} else {
		if constexpr (!ZHETAPI_CUDA)
			throw std::runtime_error("Tensor::verbose(): CUDA variant is not available");

#ifdef __CUDACC__
		// Copy to CPU
		std::vector <T> data(_shape.elements);

		// TODO: use helper...
		cudaMemcpy(
			data.data(), _array.get(),
			_shape.elements * sizeof(T),
			cudaMemcpyDeviceToHost
		);

		return verbose_str(data.data(), _shape);
#endif
	}
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
