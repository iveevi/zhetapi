#ifndef FIXED_VECTOR_H_
#define FIXED_VECTOR_H_

// C/C++ headers
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <functional>

// Engine headers
#include "vector_type.hpp"

namespace zhetapi {

// FixedVector class
template <class T, size_t N>
class FixedVector : public VectorType <T> {
	T _array[N];
public:
	FixedVector(T *, size_t = N);
	FixedVector(std::function <T (size_t)>, size_t = N);

	size_t size() const override;

	T &get(size_t) override;
	const T &get(size_t) const override;

	T &operator[](size_t) override;
	const T &operator[](size_t) const override;

	T norm() const override;
	FixedVector normalized() const;

	FixedVector &operator+=(const VectorType <T> &) override;
	FixedVector &operator-=(const VectorType <T> &) override;

	FixedVector &operator*=(const T &) override;
	FixedVector &operator/=(const T &) override;

	// Output operators
	template <class U, size_t K>
	friend std::ostream &operator<<(std::ostream &, const FixedVector <U, K> &);
};

// n is the number of elements to transfer (rest are 0)
template <class T, size_t N>
FixedVector <T, N> ::FixedVector(T *a, size_t n)
{
	for (size_t i = 0; i < n; i++)
		_array[i] = a[i];
}

template <class T, size_t N>
FixedVector <T, N> ::FixedVector(std::function <T (size_t)> ftn, size_t n)
{
	for (size_t i = 0; i < n; i++)
		_array[i] = ftn(i);
}

template <class T, size_t N>
size_t FixedVector <T, N> ::size() const
{
	return N;
}

template <class T, size_t N>
T &FixedVector <T, N> ::get(size_t i)
{
	return _array[i];
}

template <class T, size_t N>
const T &FixedVector <T, N> ::get(size_t i) const
{
	return _array[i];
}

template <class T, size_t N>
T &FixedVector <T, N> ::operator[](size_t i)
{
	return _array[i];
}

template <class T, size_t N>
const T &FixedVector <T, N> ::operator[](size_t i) const
{
	return _array[i];
}

template <class T, size_t N>
T FixedVector <T, N> ::norm() const
{
	T sq = 0;
	for (size_t i = 0; i < N; i++)
		sq += _array[i] * _array[i];
}

template <class T, size_t N>
FixedVector <T, N> FixedVector <T, N> ::normalized() const
{
	T n = norm();
	return *this/n;
}

template <class T, size_t N>
FixedVector <T, N> &FixedVector <T, N> ::operator+=(const VectorType <T> &other)
{
	assert(other.size() == N);
	for (size_t i = 0; i < N; i++)
		_array[i] += other[i];

	return *this;
}

template <class T, size_t N>
FixedVector <T, N> &FixedVector <T, N> ::operator-=(const VectorType <T> &other)
{
	assert(other.size() == N);
	for (size_t i = 0; i < N; i++)
		_array[i] -= other[i];

	return *this;
}

template <class T, size_t N>
FixedVector <T, N> &FixedVector <T, N> ::operator*=(const T &k)
{
	for (size_t i = 0; i < N; i++)
		_array[i] *= k;

	return *this;
}

template <class T, size_t N>
FixedVector <T, N> &FixedVector <T, N> ::operator/=(const T &k)
{
	assert(k != 0);
	for (size_t i = 0; i < N; i++)
		_array[i] /= k;

	return *this;
}

template <class T, size_t N>
std::ostream &operator<<(std::ostream &os, const FixedVector <T, N> &fv)
{
	os << '<';

	for (size_t i = 0; i < N; i++) {
		os << fv[i];

		if (i < N - 1)
			os << ", ";
	}

	return os << '>';
}

// FixedVector for 3D
template <class T>
class FixedVector <T, 3> : public VectorType <T> {
public:
	T x, y, z;

	FixedVector(const T & = T(), const T & = T(), const T & = T());

	FixedVector(T *, size_t = 3);
	FixedVector(std::function <T (size_t)>, size_t = 3);

	size_t size() const override;

	T &get(size_t) override;
	const T &get(size_t) const override;

	T &operator[](size_t) override;
	const T &operator[](size_t) const override;

	T norm() const override;
	FixedVector normalized() const;

	FixedVector &operator+=(const VectorType <T> &) override;
	FixedVector &operator-=(const VectorType <T> &) override;

	FixedVector &operator+=(const FixedVector &);
	FixedVector &operator-=(const FixedVector &);

	FixedVector &operator*=(const T &) override;
	FixedVector &operator/=(const T &) override;
};

template <class T>
FixedVector <T, 3> ::FixedVector(const T &xp, const T &yp, const T &zp)
		: x(xp), y(yp), z(zp) {}

template <class T>
FixedVector <T, 3> ::FixedVector(T *a, size_t n)
{
	x = (n > 0) ? a[0] : 0;
	y = (n > 1) ? a[1] : 0;
	z = (n > 2) ? a[2] : 0;
}

template <class T>
FixedVector <T, 3> ::FixedVector(std::function <T (size_t)> ftn, size_t n)
{
	x = (n > 0) ? ftn(0) : 0;
	y = (n > 1) ? ftn(1) : 0;
	z = (n > 2) ? ftn(2) : 0;
}

template <class T>
size_t FixedVector <T, 3> ::size() const
{
	return 3;
}

template <class T>
T &FixedVector <T, 3> ::get(size_t i)
{
	return (i == 0) ? x : ((i == 1) ? y : z);
}

template <class T>
const T &FixedVector <T, 3> ::get(size_t i) const
{
	return (i == 0) ? x : ((i == 1) ? y : z);
}

template <class T>
T &FixedVector <T, 3> ::operator[](size_t i)
{
	return (i == 0) ? x : ((i == 1) ? y : z);
}

template <class T>
const T &FixedVector <T, 3> ::operator[](size_t i) const
{
	return (i == 0) ? x : ((i == 1) ? y : z);
}

template <class T>
T FixedVector <T, 3> ::norm() const
{
	return std::sqrt(x * x + y * y + z * z);
}

template <class T>
FixedVector <T, 3> FixedVector <T, 3> ::normalized() const
{
	T n = norm();
	return FixedVector <T, 3> {x/n, y/n, z/n};
}

template <class T>
FixedVector <T, 3> &FixedVector <T, 3> ::operator+=(const VectorType <T> &other)
{
	assert(other.size() == 3);

	x += other[0];
	y += other[1];
	z += other[2];

	return *this;
}

template <class T>
FixedVector <T, 3> &FixedVector <T, 3> ::operator-=(const VectorType <T> &other)
{
	assert(other.size() == 3);

	x -= other[0];
	y -= other[1];
	z -= other[2];

	return *this;
}

template <class T>
FixedVector <T, 3> &FixedVector <T, 3> ::operator+=(const FixedVector &other)
{
	x += other.x;
	y += other.y;
	z += other.z;

	return *this;
}

template <class T>
FixedVector <T, 3> &FixedVector <T, 3> ::operator-=(const FixedVector &other)
{
	x -= other.x;
	y -= other.y;
	z -= other.z;

	return *this;
}

template <class T>
FixedVector <T, 3> &FixedVector <T, 3> ::operator*=(const T &k)
{
	x *= k;
	y *= k;
	z *= k;

	return *this;
}

template <class T>
FixedVector <T, 3> &FixedVector <T, 3> ::operator/=(const T &k)
{
	assert(k != 0);

	x /= k;
	y /= k;
	z /= k;

	return *this;
}

// Optimized dot product
template <class T>
T dot(const FixedVector <T, 3> &a, const FixedVector <T, 3> &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <class T>
FixedVector <T, 3> cross(const FixedVector <T, 3> &a, const FixedVector <T, 3> &b)
{
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

// FixedVector for 2D
template <class T>
class FixedVector <T, 2> : public VectorType <T> {
public:
	T x, y;

	FixedVector(const T & = T(), const T & = T());

	FixedVector(T *, size_t = 2);
	FixedVector(std::function <T (size_t)>, size_t = 2);

	size_t size() const override;

	T &get(size_t) override;
	const T &get(size_t) const override;

	T &operator[](size_t) override;
	const T &operator[](size_t) const override;

	T norm() const override;
	FixedVector normalized() const;

	FixedVector &operator+=(const VectorType <T> &) override;
	FixedVector &operator-=(const VectorType <T> &) override;

	FixedVector &operator+=(const FixedVector &);
	FixedVector &operator-=(const FixedVector &);

	FixedVector &operator*=(const T &) override;
	FixedVector &operator/=(const T &) override;
};

template <class T>
FixedVector <T, 2> ::FixedVector(const T &xp, const T &yp)
		: x(xp), y(yp) {}

template <class T>
FixedVector <T, 2> ::FixedVector(T *a, size_t n)
{
	x = (n > 0) ? a[0] : 0;
	y = (n > 1) ? a[1] : 0;
}

template <class T>
FixedVector <T, 2> ::FixedVector(std::function <T (size_t)> ftn, size_t n)
{
	x = (n > 0) ? ftn(0) : 0;
	y = (n > 1) ? ftn(1) : 0;
}

template <class T>
size_t FixedVector <T, 2> ::size() const
{
	return 2;
}

template <class T>
T &FixedVector <T, 2> ::get(size_t i)
{
	return (i == 0) ? x : y;
}

template <class T>
const T &FixedVector <T, 2> ::get(size_t i) const
{
	return (i == 0) ? x : y;
}

template <class T>
T &FixedVector <T, 2> ::operator[](size_t i)
{
	return (i == 0) ? x : y;
}

template <class T>
const T &FixedVector <T, 2> ::operator[](size_t i) const
{
	return (i == 0) ? x : y;
}


template <class T>
T FixedVector <T, 2> ::norm() const
{
	return std::sqrt(x * x + y * y);
}

template <class T>
FixedVector <T, 2> FixedVector <T, 2> ::normalized() const
{
	T n = norm();
	return FixedVector <T, 2> {x/n, y/n};
}

template <class T>
FixedVector <T, 2> &FixedVector <T, 2> ::operator+=(const VectorType <T> &other)
{
	assert(other.size() == 2);

	x += other[0];
	y += other[1];

	return *this;
}

template <class T>
FixedVector <T, 2> &FixedVector <T, 2> ::operator-=(const VectorType <T> &other)
{
	assert(other.size() == 2);

	x -= other[0];
	y -= other[1];

	return *this;
}

template <class T>
FixedVector <T, 2> &FixedVector <T, 2> ::operator+=(const FixedVector &other)
{
	x += other.x;
	y += other.y;

	return *this;
}

template <class T>
FixedVector <T, 2> &FixedVector <T, 2> ::operator-=(const FixedVector &other)
{
	x -= other.x;
	y -= other.y;

	return *this;
}

template <class T>
FixedVector <T, 2> &FixedVector <T, 2> ::operator*=(const T &k)
{
	x *= k;
	y *= k;

	return *this;
}

template <class T>
FixedVector <T, 2> &FixedVector <T, 2> ::operator/=(const T &k)
{
	assert(k != 0);

	x /= k;
	y /= k;

	return *this;
}

// Binary operations
template <class T>
FixedVector <T, 2> operator+(const FixedVector <T, 2> &a,
		const FixedVector <T, 2> &b)
{
	return {a.x + b.x, a.y + b.y};
}

template <class T>
FixedVector <T, 2> operator-(const FixedVector <T, 2> &a,
		const FixedVector <T, 2> &b)
{
	return {a.x - b.x, a.y - b.y};
}

template <class T>
FixedVector <T, 2> operator*(T k, const FixedVector <T, 2> &v)
{
	return {k * v.x, k * v.y};
}

// Optimized dot product
template <class T>
T dot(const FixedVector <T, 2> &a, const FixedVector <T, 2> &b)
{
	return a.x * b.x + a.y * b.y;
}

// Aliases
template <class T, size_t N>
using FVec = FixedVector <T, N>;

template <class T>
using Vec3 = FixedVector <T, 3>;

template <class T>
using Vec2 = FixedVector <T, 2>;

using Vec3i = Vec3 <int>;
using Vec3f = Vec3 <float>;
using Vec3d = Vec3 <double>;

using Vec2i = Vec2 <int>;
using Vec2f = Vec2 <float>;
using Vec2d = Vec2 <double>;

}

#endif
