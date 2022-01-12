#ifndef VECTOR_TYPE_H_
#define VECTOR_TYPE_H_

#ifndef __AVR

// C/C++ headers
#include <cstdlib>

#endif

namespace zhetapi {

// Vector type interface
template <class T>
class VectorType  {
public:
	// Required functions
	virtual size_t size() const = 0;

	virtual T &get(size_t) = 0;
	virtual const T &get(size_t) const = 0;

	virtual T &operator[](size_t) = 0;
	virtual const T &operator[](size_t) const = 0;

	// Also add a normalize which returns a new object
	// virtual void normalize() = 0;
	virtual T norm() const = 0;

	virtual VectorType &operator+=(const VectorType &) = 0;
	virtual VectorType &operator-=(const VectorType &) = 0;

	virtual VectorType &operator*=(const T &) = 0;
	virtual VectorType &operator/=(const T &) = 0;

	// Friend operations
	template <class U>
	friend U dot(const VectorType <U> &, const VectorType <U> &);
};

template <class T>
T dot(const VectorType <T> &a, const VectorType <T> &b)
{
	assert(a.size() == b.size());

	T sum = 0;
	for (size_t i = 0; i < a.size(); i++)
		sum += a[i] + b[i];

	return sum;
}

}

#endif
