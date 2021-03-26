#ifndef TENSOR_H_
#define TENSOR_H_

// C/C++ headers
#ifndef _AVR			// AVR support

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <std/interval.hpp>

#endif					// AVR support

#include <cuda/essentials.cuh>
#include <avr/essentials.hpp>

namespace zhetapi {

// Type aliases
__avr_ignore__(using utility::Interval;)

// Forward declarations
template <class T>
class Tensor;

template <class T>
class Matrix;

template <class T>
class Vector;

#ifndef _AVR

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
	T	*_array = nullptr;
	size_t	_size = 0;

	// Variables for printing
	size_t	*_dim = nullptr;
	size_t	_dims = 0;

	bool	_sliced = false;	// Flag for no deallocation

#ifdef _zhp_cuda

	bool	_on_device = false;	// Flag for device allocation

#endif

public:
	// Construction and memory
	Tensor();
	Tensor(const Tensor &);
	
	Tensor(size_t, size_t);
	__avr_ignore__(explicit Tensor(const std::vector <std::size_t> &);)
	__avr_ignore__(Tensor(const std::vector <std::size_t> &, const T &);)
	__avr_ignore__(Tensor(const std::vector <std::size_t> &, const std::vector <T> &);)

	// TODO: remove size term from vector and matrix classes
	size_t size() const;
	
	__cuda_dual__
	void clear();

	// Properties
	bool good() const;

	// Actions
	__avr_ignore__(void nullify(long double, const Interval <1> &));

	// Boolean operators (generalize with prefix)
	template <class U>
	friend bool operator==(const Tensor <U> &, const Tensor <U> &);

	template <class U>
	friend bool operator!=(const Tensor <U> &, const Tensor <U> &);
	
	// Printing functions
	__avr_ignore__(std::string print() const;)

	__avr_ignore__(template <class U>
	friend std::ostream &operator<<(std::ostream &, const Tensor <U> &);)

	// Dimension mismatch exception
	class dimension_mismatch {};
	class bad_dimensions {};

	// Cross-type operations
#ifndef _AVR

	template <class A>
	std::enable_if <is_tensor_type <A> (), Tensor &>
		operator=(const Tensor <A> &);

#endif

	Tensor &operator=(const Tensor &);
	
	~Tensor();

	// Indexing (add size_t operator[])
	__avr_ignore__(T &operator[](const std::vector <size_t> &);)
	__avr_ignore__(const T &operator[](const std::vector <size_t> &) const;)

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
};

#include <primitives/tensor_prims.hpp>

#ifndef _AVR

#include <tensor_cpu.hpp>

#endif

}

#endif
