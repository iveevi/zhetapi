#ifndef FIXED_VECTOR_H_
#define FIXED_VECTOR_H_

// C/C++ headers
#include <cmath>
#include <cassert>
#include <cstdlib>

// Engine headers
#include "vector_type.hpp"

namespace zhetapi {

// FixedVector class
template <class T, size_t N>
class FixedVector : public VectorType <T> {
	T _array[N];
public:
	size_t size() const override;

	T &get(size_t) override;
	const T &get(size_t) const override;

	T &operator[](size_t) override;
	const T &operator[](size_t) const override;

	FixedVector &operator+=(const VectorType <T> &) override;
	FixedVector &operator-=(const VectorType <T> &) override;

	FixedVector &operator*=(const T &) override;
	FixedVector &operator/=(const T &) override;
};

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

// FixedVector for 3D
template <class T>
class FixedVector <T, 3> : public VectorType <T> {
public:
	T x, y, z;

	FixedVector(const T & = T(), const T & = T(), const T & = T());

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

	size_t size() const override;

	T &get(size_t) override;
	const T &get(size_t) const override;

	T &operator[](size_t) override;
	const T &operator[](size_t) const override;

	FixedVector &operator+=(const VectorType <T> &) override;
	FixedVector &operator-=(const VectorType <T> &) override;

	FixedVector &operator+=(const FixedVector &);
	FixedVector &operator-=(const FixedVector &);

	FixedVector &operator*=(const T &) override;
	FixedVector &operator/=(const T &) override;
};

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
