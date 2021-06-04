#ifndef MATRIX_H_
#define MATRIX_H_

// C/C++ headers
#ifndef __AVR			// AVR support

#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

#endif					// AVR support

// Engine headers
#ifdef ZHP_CUDA

#include "cuda/tensor.cuh"

#else

#include "tensor.hpp"

#endif

#include "cuda/essentials.cuh"

// Redeclare minor as a matrix operation
#ifdef minor

#undef minor

#endif

// In function initialization
#define inline_init_mat(mat, rs, cs)			\
	Matrix <T> mat;					\
							\
	mat._rows = rs;				\
	mat._cols = cs;				\
	mat._size = rs * cs;				\
							\
	mat._array = new T[rs * cs];			\
							\
	memset(mat._array, 0, rs * cs * sizeof(T));	\
							\
	mat._dims = 2;					\
							\
	mat._dim = new size_t[2];			\
							\
	mat._dim[0] = rs;				\
	mat._dim[1] = cs;

namespace zhetapi {
template <class T>
class Vector;

/**
 * @brief A matrix with components of type T.
 *
 * @tparam T the type of each component.
 */
template <class T>
class Matrix : public Tensor <T> {
protected:
	// TODO: Remove later
	size_t  _rows	= 0;
	size_t  _cols	= 0;
public:
	__cuda_dual__ Matrix();
	__cuda_dual__ Matrix(const Matrix &);
	__cuda_dual__ Matrix(const Vector <T> &);

	// Scaled
	__cuda_dual__ Matrix(const Matrix &, T);

	__cuda_dual__ Matrix(size_t, size_t, T = T());

	// Lambda constructors
	AVR_SWITCH(
		Matrix(size_t, size_t, T (*)(size_t)),
		Matrix(size_t, size_t, std::function <T (size_t)>)
	);

	AVR_SWITCH(
		Matrix(size_t, size_t, T *(*)(size_t)),
		Matrix(size_t, size_t, std::function <T *(size_t)>)
	);
	
	AVR_SWITCH(
		Matrix(size_t, size_t, T (*)(size_t, size_t)),
		Matrix(size_t, size_t, std::function <T (size_t, size_t)>)
	);

	AVR_SWITCH(
		Matrix(size_t, size_t, T *(*)(size_t, size_t)),
		Matrix(size_t, size_t, std::function <T *(size_t, size_t)>)
	);

	AVR_IGNORE(Matrix(const std::vector <T> &));
	AVR_IGNORE(Matrix(const std::vector <Vector <T>> &));
	AVR_IGNORE(Matrix(const std::vector <std::vector <T>> &));
	AVR_IGNORE(Matrix(const std::initializer_list <Vector <T>> &));
	AVR_IGNORE(Matrix(const std::initializer_list <std::initializer_list <T>> &));

	__cuda_dual__
	Matrix(size_t, size_t, T *, bool = true);

	inline T &get(size_t, size_t);
	inline const T &get(size_t, size_t) const;
	
	T norm() const;

	void resize(size_t, size_t);

	psize_t get_dimensions() const;

	Matrix slice(const psize_t &, const psize_t &) const;

	void set(size_t, size_t, T);

	const T &get(size_t, size_t) const;

	Vector <T> get_column(size_t) const;

	// Rading from a binary file (TODO: unignore later)
	AVR_IGNORE(void write(std::ofstream &) const;)
	AVR_IGNORE(void read(std::ifstream &);)

	// Concatenating matrices
	Matrix append_above(const Matrix &);
	Matrix append_below(const Matrix &);

	Matrix append_left(const Matrix &);
	Matrix append_right(const Matrix &);

	void operator*=(const Matrix &);
	void operator/=(const Matrix &);

	// Row operations
	void add_rows(size_t, size_t, T);
	
	void swap_rows(size_t, size_t);

	void multiply_row(size_t, T);

	void pow(const T &);

	// Miscellanious opertions
	AVR_IGNORE(void randomize(std::function <T ()>);)
	
	__cuda_dual__
	void row_shur(const Vector <T> &);
	
	__cuda_dual__
	void stable_shur(const Matrix <T> &);

	__cuda_dual__
	void stable_shur_relaxed(const Matrix <T> &);

	// Values
	T determinant() const;

	T minor(size_t, size_t) const;
	T minor(const psize_t &) const;

	T cofactor(size_t, size_t) const;
	T cofactor(const psize_t &) const;

	Matrix inverse() const;
	Matrix adjugate() const;
	Matrix cofactor() const;

	// Property checkers
	bool is_symmetric(const T & = EPSILON) const;
	bool is_diagonal(const T & = EPSILON) const;
	bool is_identity(const T & = EPSILON) const;
	bool is_orthogonal(const T & = EPSILON) const;
	bool is_lower_triangular(const T & = EPSILON) const;
	bool is_upper_triangular(const T & = EPSILON) const;

	AVR_SWITCH(
		String display() const,
		std::string display() const
	);
	
#ifndef __AVR

	template <class U>
	friend std::ostream &operator<<(std::ostream &, const Matrix <U> &);

#endif

	// Special matrix generation
	static Matrix identity(size_t);

	// Miscellaneous functions
	template <class U>
	friend Vector <U> apt_and_mult(const Matrix <U> &, const Vector <U> &); 
	
	template <class U>
	friend Vector <U> rmt_and_mult(const Matrix <U> &, const Vector <U> &);

	template <class U>
	friend Matrix <U> vvt_mult(const Vector <U> &, const Vector <U> &); 

	class dimension_mismatch {};
protected:
	// TODO: Looks ugly here
	T determinant(const Matrix &) const;
public:
	const Matrix &operator=(const Matrix &);

	T *operator[](size_t);
	const T *operator[](size_t) const;

	size_t get_rows() const;
	size_t get_cols() const;
	
	Matrix transpose() const;

	void operator+=(const Matrix &);
	void operator-=(const Matrix &);
	
	void operator*=(const T &);
	void operator/=(const T &);

	// Matrix and matrix operations
	template <class U>
	friend Matrix <U> operator+(const Matrix <U> &, const Matrix <U> &);
	
	template <class U>
	friend Matrix <U> operator-(const Matrix <U> &, const Matrix <U> &);
	
	// template <class U>
	// friend Matrix <U> operator*(const Matrix <U> &, const Matrix <U> &);
	
	// Heterogenous multiplication
	template <class U, class V>
	friend Matrix <U> operator*(const Matrix <U> &, const Matrix <V> &);
	
	template <class U>
	friend Matrix <U> operator*(const Matrix <U> &, const U &);
	
	template <class U>
	friend Matrix <U> operator*(const U &, const Matrix <U> &);
	
	template <class U>
	friend Matrix <U> operator/(const Matrix <U> &, const U &);
	
	template <class U>
	friend Matrix <U> operator/(const U &, const Matrix <U> &);

	template <class U>
	friend bool operator==(const Matrix <U> &, const Matrix <U> &);
	
	// Miscellaneous operations
	template <class U>
	friend Matrix <U> shur(const Matrix <U> &, const Matrix <U> &);
	
	template <class U>
	friend Matrix <U> inv_shur(const Matrix <U> &, const Matrix <U> &);

	template <class A, class B, class C>
	friend Matrix <A> fma(const Matrix <A> &, const Matrix <B> &, const Matrix <C> &);

	template <class A, class B, class C>
	friend Matrix <A> fmak(const Matrix <A> &, const Matrix <B> &, const Matrix <C> &, A, A);

	static T EPSILON;
};

}

// TODO use _CUDACC_ instead of _zhp_cuda and make _cuda files
#include "primitives/matrix_prims.hpp"

#ifndef __AVR

#include "matrix_cpu.hpp"

#endif

#endif
