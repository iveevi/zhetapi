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

// Engine headers
#include "std/interval.hpp"

// Engine headers
#include "cuda/essentials.cuh"
#include "avr/essentials.hpp"

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

#ifdef __CUDACC__

class NVArena;

#endif

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

	AVR_IGNORE(explicit Tensor(const std::vector <std::size_t> &));
	AVR_IGNORE(Tensor(const std::vector <std::size_t> &, const T &));
	AVR_IGNORE(Tensor(const std::vector <std::size_t> &, const std::vector <T> &));

	// Indexing
	Tensor <T> operator[](size_t);

	// TODO: iterators
	// TODO: this type of indexing is very tedious with the [], use anotehr
	// method like .get(...)
	AVR_IGNORE(T &operator[](const std::vector <size_t> &));
	AVR_IGNORE(const T &operator[](const std::vector <size_t> &) const);

	// TODO: remove size term from vector and matrix classes
	__cuda_dual__ size_t size() const;
	// __cuda_dual__ size_t dimensions() const;
	__cuda_dual__ size_t dim_size(size_t) const;
	__cuda_dual__ size_t safe_dim_size(size_t) const;

	// TODO: refactor to shape()
	std::vector <size_t> dimensions() const;

	// TODO: private?
	__cuda_dual__
	void clear();

	// Properties
	bool good() const;

	// Actions
	AVR_IGNORE(void nullify(long double, const utility::Interval <1> &));

	// Boolean operators (generalize with prefix)
	template <class U>
	friend bool operator==(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend bool operator!=(const Tensor <U> &, const Tensor <U> &);

	// Printing functions
	AVR_IGNORE(std::string print() const;)

	AVR_IGNORE(template <class U>
	friend std::ostream &operator<<(std::ostream &, const Tensor <U> &));

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

	// Apply a function to each element of the tensor
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

// Applying element-wise transformations
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
	Tensor <T> c(a.dimensions());
	for (int i = 0; i < a._size; i++)
		c._array[i] = a._array[i] * b._array[i];
	return c;
}

template <class T>
Tensor <T> divide(const Tensor <T> &a, const Tensor <T> &b)
{
	// TODO: check dimensions (make a helper function for this)
	Tensor <T> c(a.dimensions());
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

}

#include "primitives/tensor_prims.hpp"

// TODO: remove this branch, screw AVR
#ifndef __AVR

#include "tensor_cpu.hpp"

#endif

#endif
