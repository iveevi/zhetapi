#ifndef ELEMENT_H_
#define ELEMENT_H_

// C/C++ headers
#include <cmath>
#include <functional>

// Engine headers
#ifdef ZHP_CUDA

#include <cuda/matrix.cuh>

#else

#include <matrix.hpp>

#endif

#include <cuda/essentials.cuh>

namespace zhetapi {

/**
 * @brief Represents a vector in mathematics, on the scalar field corresponding
 * to T. Derived from the matrix class.
 * */
template <class T>
class Vector : public Matrix <T> {
public:	
	Vector(const ::std::vector <T> &);
	Vector(const ::std::initializer_list <T> &);

	template <class A>
	Vector(A);

	T arg() const;

	// Min and max index
	size_t imin() const;
	size_t imax() const;

	// Normalization
	void normalize();

	Vector normalized() const;

	// Vector operations
	template <class U>
	friend Vector <U> cross(const Vector <U> &, const Vector <U> &);

	// Static methods
	static Vector one(size_t);
	static Vector rarg(double, double);
	
#ifndef ZHP_CUDA
	
	Vector();
	Vector(const Vector &);
	Vector(const Matrix <T> &);
	
	Vector(size_t, T);
	Vector(size_t, T *);

	Vector(size_t, ::std::function <T (size_t)>);
	Vector(size_t, ::std::function <T *(size_t)>);

	Vector &operator=(const Vector &);
	Vector &operator=(const Matrix <T> &);

	T &operator[](size_t);
	const T &operator[](size_t) const;
	
	size_t size() const;

	void operator+=(const Vector &);
	void operator-=(const Vector &);
	
	Vector append_above(const T &);
	Vector append_below(const T &);

	Vector remove_top();
	Vector remove_bottom();
	
	T norm() const;
	
	template <class U>
	friend U inner(const Vector <U> &, const Vector <U> &);

#else
	
	__host__ __device__
	Vector();
	
	__host__ __device__
	Vector(const Vector &);
	
	__host__ __device__
	Vector(const Matrix <T> &);
	
	__host__ __device__
	Vector(size_t, T);
	
	__host__ __device__
	Vector(size_t, T *, bool = false);
	
	template <class F>
	__host__ __device__
	Vector(size_t, F);

	__host__ __device__
	Vector &operator=(const Vector &);
	
	__host__ __device__
	Vector &operator=(const Matrix <T> &);
	
	// Other memory concerned operations
	void copy_to_device(const Vector <T> &);

	void transfer_from_device(Vector <T> &);

	PREFIX
	T *whole() const {
		return this->__array;
	}

	__host__ __device__
	T &operator[](size_t);

	__host__ __device__
	const T &operator[](size_t) const;
	
	__host__ __device__
	size_t size() const;

	__host__ __device__
	void operator+=(const Vector &);
	
	__host__ __device__
	void operator-=(const Vector &);
	
	__host__ __device__
	Vector append_above(const T &);
	
	__host__ __device__
	Vector append_below(const T &);

	__host__ __device__
	Vector remove_top();
	
	__host__ __device__
	Vector remove_bottom();

	__host__ __device__
	T norm() const;

	template <class U>
	__host__ __device__
	friend Vector <U> operator-(const Vector <U> &, const Vector <U> &);

	template <class U>
	__host__ __device__
	friend Vector <U> operator*(const Vector <U> &, const U &);

	template <class U>
	__host__ __device__
	friend Vector <U> operator*(const U &, const Vector <U> &);
	
	template <class U>
	__host__ __device__
	friend U inner(const Vector <U> &, const Vector <U> &);

#endif

};

template <class T>
Vector <T> ::Vector(const ::std::vector <T> &ref) : Matrix <T> (ref) {}

template <class T>
Vector <T> ::Vector(const ::std::initializer_list <T> &ref)
	: Vector(::std::vector <T> (ref)) {}

template <class T>
template <class A>
Vector <T> ::Vector(A x)
{
}

template <class T>
T Vector <T> ::arg() const
{
	return atan2((*this)[1], (*this)[0]);
}

template <class T>
size_t Vector <T> ::imin() const
{
	size_t i = 0;

	for (size_t j = 1; j < this->__size; j++) {
		if (this->__array[i] > this->__array[j])
			i = j;
	}

	return i;
}

template <class T>
size_t Vector <T> ::imax() const
{
	size_t i = 0;

	for (size_t j = 1; j < this->__size; j++) {
		if (this->__array[i] < this->__array[j])
			i = j;
	}

	return i;
}

template <class T>
void Vector <T> ::normalize()
{
	T dt = norm();

	for (size_t i = 0; i < size(); i++)
		(*this)[i] /= dt;
}

template <class T>
Vector <T> Vector <T> ::normalized() const
{
	::std::vector <T> out;

	T dt = norm();

	for (size_t i = 0; i < size(); i++)
		out.push_back((*this)[i]/dt);

	return Vector(out);
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

// Non-member functions
template <class T>
T cross(const Vector <T> &a, const Vector <T> &b)
{
	assert(a.__size == b.size() == 3);

	return {
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]
	};
}

// Static methods
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

#ifndef ZHP_CUDA

template <class T>
Vector <T> ::Vector() : Matrix <T> () {}

template <class T>
Vector <T> ::Vector(const Vector &other) : Matrix <T> (other.size(), 1, T())
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] = other.__array[i];
}

template <class T>
Vector <T> ::Vector(const Matrix <T> &other) : Matrix <T> (other.get_rows(), 1, T())
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] = other[0][i];
}

template <class T>
Vector <T> ::Vector(size_t rs, T def) : Matrix <T> (rs, 1, def) {}

// FIXME: Delegate Matrix constructor
template <class T>
Vector <T> ::Vector(size_t rs, T *ref) : Matrix <T> (rs, 1, ref) {}

template <class T>
Vector <T> ::Vector(size_t rs, ::std::function <T (size_t)> gen)
	: Matrix <T> (rs, 1, gen) {}

template <class T>
Vector <T> ::Vector(size_t rs, ::std::function <T *(size_t)> gen)
	: Matrix <T> (rs, 1, gen) {}

template <class T>
Vector <T> &Vector <T> ::operator=(const Vector <T> &other)
{
	if (this != &other) {
		delete[] this->__array;
		delete[] this->__dim;

		this->__array = new T[other.__size];
		this->__rows = other.__rows;
		this->__cols = other.__cols;

		this->__size = other.__size;
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = other.__array[i];
		
		this->__dims = 1;
		this->__dim = new size_t[1];

		this->__dim[0] = this->__size;
	}

	return *this;
}

template <class T>
Vector <T> &Vector <T> ::operator=(const Matrix <T> &other)
{
	if (this != &other) {
		*this = Vector(other.get_rows(), T());

		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = other[0][i];
	}

	return *this;
}

template <class T>
T &Vector <T> ::operator[](size_t i)
{
	return this->__array[i];
}

template <class T>
const T &Vector <T> ::operator[](size_t i) const
{
	return this->__array[i];
}

template <class T>
size_t Vector <T> ::size() const
{
	return this->__size;
}

template <class T>
void Vector <T> ::operator+=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] += a.__array[i];
}

template <class T>
void Vector <T> ::operator-=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] -= a.__array[i];
}

template <class T>
Vector <T> Vector <T> ::append_above(const T &x)
{
	size_t t_sz = size();

	::std::vector <T> total {x};

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	return Vector(total);
}

template <class T>
Vector <T> Vector <T> ::append_below(const T &x)
{
	size_t t_sz = size();

	::std::vector <T> total;

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	total.push_back(x);

	return Vector(total);
}

template <class T>
Vector <T> Vector <T> ::remove_top()
{
	size_t t_sz = size();

	::std::vector <T> total;
	for (size_t i = 1; i < t_sz; i++)
		total.push_back((*this)[i]);

	return Vector(total);
}

template <class T>
Vector <T> Vector <T> ::remove_bottom()
{
	size_t t_sz = size();

	::std::vector <T> total;
	for (size_t i = 0; i < t_sz - 1; i++)
		total.push_back((*this)[i]);

	return Vector(total);
}

template <class T>
T Vector <T> ::norm() const
{
	return sqrt(inner(*this, *this));
}

template <class T>
T inner(const Vector <T> &a, const Vector <T> &b)
{
	T acc = 0;

	assert(a.size() == b.size());
	for (size_t i = 0; i < a.__size; i++)
		acc += a[i] * b[i];

	return acc;
}

#endif

}

#endif
