#ifndef TENSOR_H_
#define TENSOR_H_

// C/C++ headers
#ifndef __AVR			// Does not support AVR

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// Engine headers
#include "std/interval.hpp"

#endif				// Does not support AVR

// Engine headers
#include "cuda/essentials.cuh"
#include "avr/essentials.hpp"

namespace zhetapi {

// Type aliases
// AVR_IGNORE(using utility::Interval;)

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

// TODO: is this even needed
#ifndef __AVR

// Tensor_type operations
template <class T>
struct Tensor_type : std::false_type {};

template <class T>
struct Tensor_type <Tensor <T>> : std::true_type {};

template <class T>
bool is_tensor_type()
{
	return Tensor_type <T> ::value;
}

#endif

// Tensor class
template <class T>
class Tensor {
protected:
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
	// Essential constructors
	Tensor();
	Tensor(const Tensor &);

	template <class A>
	Tensor(const Tensor <A> &);

	Tensor(size_t, size_t);
	Tensor(size_t, size_t *, size_t, T *, bool = true);

	AVR_IGNORE(explicit Tensor(const std::vector <std::size_t> &));
	AVR_IGNORE(Tensor(const std::vector <std::size_t> &, const T &));
	AVR_IGNORE(Tensor(const std::vector <std::size_t> &, const std::vector <T> &));

	// Indexing
	Tensor <T> operator[](size_t);

	AVR_IGNORE(T &operator[](const std::vector <size_t> &));
	AVR_IGNORE(const T &operator[](const std::vector <size_t> &) const);

	// TODO: remove size term from vector and matrix classes
	__cuda_dual__ size_t size() const;
	__cuda_dual__ size_t dimensions() const;
	__cuda_dual__ size_t dim_size(size_t) const;
	__cuda_dual__ size_t safe_dim_size(size_t) const;

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

};

}

#include "primitives/tensor_prims.hpp"

#ifndef __AVR

#include "tensor_cpu.hpp"

#endif

#endif
