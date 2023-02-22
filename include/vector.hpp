#ifndef VECTOR_H_
#define VECTOR_H_

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <cmath>
#include <functional>

#include "matrix.hpp"
#include "field.hpp"

namespace zhetapi {

// TODO: reorganize and derive from vectortype

/**
 * @brief Represents a vector whose components are of type T.
 *
 * @tparam T the type of each component.
 */
template <class T>
class Vector : public Tensor <T>, Field <T, Vector <T>> {
public:
	Vector();
	Vector(const Vector &);
	Vector(const Tensor <T> &);

	Vector(size_t);
	Vector(size_t, T);

	// Lambda constructors
	Vector(size_t, std::function <T (size_t)>);
	Vector(size_t, std::function <T *(size_t)>);

	Vector(const std::vector <T> &);
	Vector(const std::initializer_list <T> &);

	// Cross-type operations
	template <class A>
	explicit Vector(const Vector <A> &);

	// Assignment
	Vector &operator=(const Vector &);
	Vector &operator=(const Matrix <T> &);

	/* Indexing
	inline T &get(size_t);
	inline const T &get(size_t) const; */

	inline T &operator[](size_t);
	inline const T &operator[](size_t) const;

	// Transpose
	Matrix <T> transpose() const {
		return Matrix <T> (*this).transpose();
	}

	// Direction of the vector (radians)
	T arg() const;

	// Min and max value
	T min() const;
	T max() const;

	// Min and max index
	size_t imin() const;
	size_t imax() const;

	// Functions
	Vector operator()(std::function <T (T)>);
	T sum(std::function <T (T)>);
	T product(std::function <T (T)>);

	// TODO: operator(), product and sum

	// Modifiers
	Vector append_above(const T &) const;
	Vector append_below(const T &);

	Vector remove_top();
	Vector remove_bottom();

	// Normalization
	void normalize();

	// Maybe a different name (is the similarity justifiable?)
	Vector normalized() const;

	// Arithmetic operators
	void operator+=(const Vector &);
	void operator-=(const Vector &);

	// Static methods
	static Vector one(size_t);
	static Vector rarg(double, double);

	// Vector operations
	template <class F, class U>
	friend U max(F, const Vector <U> &);

	template <class F, class U>
	friend U min(F, const Vector <U> &);

	template <class F, class U>
	friend U argmax(F, const Vector <U> &);

	template <class F, class U>
	friend U argmin(F, const Vector <U> &);

	template <class U>
	friend Vector <U> cross(const Vector <U> &, const Vector <U> &);

	// Vector concatenation
	template <class U>
	friend Vector <U> concat(const Vector <U> &, const Vector <U> &);

	template <class U>
	friend U inner(const Vector <U> &, const Vector <U> &);

	// Heterogenous inner product (assumes first underlying type)
	template <class U, class V>
	friend U inner(const Vector <U> &, const Vector <V> &);

	// Exceptions (put in matrix later)
	class index_out_of_bounds {};
};

template <class T>
Vector <T> ::Vector() : Tensor <T> () {}

template <class T>
Vector <T> ::Vector(const Vector <T> &other)
		: Tensor <T> (other) {}

template <class T>
Vector <T> ::Vector(const Tensor <T> &other)
		: Tensor <T> (other)
{
	this->reshape({other.size()});
}

template <class T>
Vector <T> ::Vector(size_t len)
		: Tensor <T> (typename Tensor <T> ::shape_type {len}) {}

template <class T>
Vector <T> ::Vector(size_t len, T def)
		: Tensor <T> (typename Tensor <T> ::shape_type {len}, def) {}

// Assignment operators
template <class T>
Vector <T> &Vector <T> ::operator=(const Vector <T> &other)
{
	if (this != &other)
		Tensor <T> ::operator=(other);
	return *this;
}

/**
 * @brief Indexing operator.
 *
 * @param i the specified index.
 *
 * @return the \f$i\f$th component of the vector.
 */
template <class T>
inline T &Vector <T> ::operator[](size_t i)
{
	return this->_array[i];
}

/**
 * @brief Indexing operator.
 *
 * @param i the specified index.
 *
 * @return the \f$i\f$th component of the vector.
 */
template <class T>
inline const T &Vector <T> ::operator[](size_t i) const
{
	return this->_array[i];
}

/**
 * @brief The index of the smallest component: essentially argmin.
 *
 * @return the index of the smallest component.
 */
template <class T>
size_t Vector <T> ::imin() const
{
	size_t i = 0;

	for (size_t j = 1; j < this->size(); j++) {
		if (this->_array[i] > this->_array[j])
			i = j;
	}

	return i;
}

/**
 * @brief The index of the largest component: essentially argmax.
 *
 * @return the index of the largest component.
 */
template <class T>
size_t Vector <T> ::imax() const
{
	size_t i = 0;

	for (size_t j = 1; j < this->size(); j++) {
		if (this->_array[i] < this->_array[j])
			i = j;
	}

	return i;
}

/**
 * @brief Normalizes the components of the vector (the modified vector will have
 * unit length).
 */
template <class T>
void Vector <T> ::normalize()
{
	T dt = this->norm();

	for (size_t i = 0; i < this->size(); i++)
		(*this)[i] /= dt;
}

// TODO: rename these functions (or add): they imply modification (also add const)
template <class T>
Vector <T> Vector <T> ::append_above(const T &x) const
{
	Vector <T> v(this->size() + 1);

	v[0] = x;
	for (size_t i = 0; i < this->size(); i++)
		v[i + 1] = this->_array[i];
	return v;
}

template <class T>
Vector <T> Vector <T> ::append_below(const T &x)
{
	Vector <T> v(this->size() + 1);

	for (size_t i = 0; i < this->size(); i++)
		v[i] = this->_array[i];
	v[this->size()] = x;
	return v;
}

// TODO: replace with slices
template <class T>
Vector <T> Vector <T> ::remove_top()
{
	Vector <T> v(this->size() - 1);
	for (size_t i = 1; i < this->size(); i++)
		v[i - 1] = this->_array[i];
	return v;
}

// TODO: replace with slices
template <class T>
Vector <T> Vector <T> ::remove_bottom()
{
	Vector <T> v(this->size() - 1);
	for (size_t i = 0; i < this->size() - 1; i++)
		v[i] = this->_array[i];
	return v;
}

// Non-member operators
template <class T>
Vector <T> operator+(const Vector <T> &a, const Vector <T> &b)
{
	Vector <T> out = a;

	out += b;

	return out;
}

template <class T>
Vector <T> operator-(const Vector <T> &a, const Vector <T> &b)
{
	Vector <T> out = a;

	out -= b;

	return out;
}

template <class T>
Vector <T> operator*(const Vector <T> &a, const T &b)
{
	Vector <T> out = a;

	for (size_t i = 0; i < a.size(); i++)
		out[i] *= b;

	return out;
}

template <class T>
Vector <T> operator*(const T &b, const Vector <T> &a)
{
	Vector <T> out = a;

	for (size_t i = 0; i < a.size(); i++)
		out[i] *= b;

	return out;
}

template <class T>
Vector <T> operator/(const Vector <T> &a, const T &b)
{
	Vector <T> out = a;

	for (size_t i = 0; i < a.size(); i++)
		out[i] /= b;

	return out;
}

template <class T>
Vector <T> operator/(const T &b, const Vector <T> &a)
{
	Vector <T> out = a;

	for (size_t i = 0; i < a.size(); i++)
		out[i] /= b;

	return out;
}

// Static methods
// TODO: remove some...
template <class T>
Vector <T> Vector <T> ::one(size_t size)
{
	return Vector <T> (size, T(1));
}

template <class T>
Vector <T> Vector <T> ::rarg(double r, double theta)
{
	return Vector <T> {r * cos(theta), r * sin(theta)};
}

// Non-member functions
template <class F, class T>
T max(F ftn, const Vector <T> &values)
{
	T max = ftn(values[0]);

	size_t n = values.size();
	for (size_t i = 1; i < n; i++) {
		T k = ftn(values[i]);
		if (k > max)
			max = k;
	}

	return max;
}

template <class F, class T>
T min(F ftn, const Vector <T> &values)
{
	T min = ftn(values[0]);

	size_t n = values.size();
	for (size_t i = 1; i < n; i++) {
		T k = ftn(values[i]);
		if (k < min)
			min = k;
	}

	return min;
}

template <class F, class T>
T argmax(F ftn, const Vector <T> &values)
{
	T max = values[0];

	size_t n = values.size();
	for (size_t i = 1; i < n; i++) {
		if (ftn(values[i]) > ftn(max))
			max = values[i];
	}

	return max;
}

template <class F, class T>
T argmin(F ftn, const Vector <T> &values)
{
	T min = values[0];

	size_t n = values.size();
	for (size_t i = 1; i < n; i++) {
		if (ftn(values[i]) < ftn(min))
			min = values[i];
	}

	return min;
}

template <class T>
Vector <T> cross(const Vector <T> &a, const Vector <T> &b)
{
	// Switch between 2 and 3
	assert((a._size == 3) && (a._size == 3));

	return Vector <T> {
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]
	};
}

template <class T>
Vector <T> concat(const Vector <T> &a, const Vector <T> &b)
{
	Vector <T> out(a.size() + b.size());
	for (size_t i = 0; i < a.size(); i++)
		out[i] = a[i];

	for (size_t i = 0; i < b.size(); i++)
		out[a.size() + i] = b[i];

	return out;
}

template <class T, class ... U>
Vector <T> concat(const Vector <T> &a, const Vector <T> &b, U ... args)
{
	return concat(concat(a, b), args...);
}

template <class T>
T inner(const Vector <T> &a, const Vector <T> &b)
{
	T acc = 0;

	assert(a._shape == b._shape);
	for (size_t i = 0; i < a.size(); i++)
		acc += a[i] * b[i];

	return acc;
}

template <class T, class U>
T inner(const Vector <T> &a, const Vector <U> &b)
{
	T acc = 0;

	assert(a._shape == b._shape);
	for (size_t i = 0; i < a.size(); i++)
		acc += (T) (a[i] * b[i]);	// Cast the result

	return acc;
}

/* Externally defined methods
template <class T>
Vector <T> Tensor <T> ::cast_to_vector() const
{
	// Return a slice-vector
	return Vector <T> (_size, _array);
} */

/**
 * @brief Constructs a vector out of a list of components.
 *
 * @param ref the list of components.
 */
template <class T>
Vector <T> ::Vector(const std::vector <T> &ref)
		: Tensor <T> ({ref.size()}, ref) {}

/**
 * @brief Constructs a vector out of a list of components.
 *
 * @param ref the list of components.
 */
template <class T>
Vector <T> ::Vector(const std::initializer_list <T> &ref)
		: Vector(std::vector <T> (ref)) {}

/**
 * @brief Size constructor. Each component is evaluated from a function which
 * depends on the index.
 *
 * @param rs the number of rows (size) of the vector.
 * @param gen a pointer to the function that generates the coefficients.
 */
template <class T>
Vector <T> ::Vector(size_t rs, std::function <T (size_t)> gen)
	        : Tensor <T> ({rs}, gen) {}

/**
 * @brief Size constructor. Each component is evaluated from a function which
 * depends on the index.
 *
 * @param rs the number of rows (size) of the vector.
 * @param gen a pointer to the function that generates pointers to the
 * coefficients.
 */
template <class T>
Vector <T> ::Vector(size_t rs, std::function <T *(size_t)> gen)
	        : Matrix <T> ({rs}, gen) {}

/**
 * @brief Heterogenous copy constructor.
 *
 * @param other the reference vector (to be copied from).
 */
template <class T>
template <class A>
Vector <T> ::Vector(const Vector <A> &other)
{
	// TODO: Add a new function for this
	// TODO: put this function into primitives
	// TODO: use member initializer list
	this->_array = new T[other.size()];
	this->size() = other.size();
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] = other[i];

	this->_dims = 1;
	this->_dim = new size_t[1];
	this->_dim[0] = this->size();
}

template <class T>
Vector <T> Vector <T> ::operator()(std::function <T (T)> ftn)
{
	return Vector <T> (this->size(),
		[&](size_t i) {
			return ftn(this->_array[i]);
		}
	);
}

template <class T>
T Vector <T> ::sum(std::function <T (T)> ftn)
{
	T s = 0;
	for (size_t i = 0; i < this->size(); i++)
		s += ftn(this->_array[i]);

	return s;
}

template <class T>
T Vector <T> ::product(std::function <T (T)> ftn)
{
	T p = 1;
	for (size_t i = 0; i < this->size(); i++)
		p *= ftn(this->_array[i]);

	return p;
}

/**
 * @brief Returns a vector with normalized components (length of 1). The
 * direction is preserved, as with normalization.
 *
 * @return The normalized vector.
 */
template <class T>
Vector <T> Vector <T> ::normalized() const
{
	std::vector <T> out;

	T dt = this->length();

	for (size_t i = 0; i < this->size(); i++)
		out.push_back((*this)[i]/dt);

	return Vector(out);
}

/**
 * @brief Add and assignment operator.
 *
 * TODO: Needs to return itself
 *
 * @param the vector that will be added to this.
 */
template <class T>
void Vector <T> ::operator+=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] += a._array[i];
}

/**
 * @brief Subtract and assignment operator.
 *
 * TODO: Needs to return itself
 *
 * @param the vector that will be subtracted from this.
 */
template <class T>
void Vector <T> ::operator-=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] -= a._array[i];
}

}

#endif
