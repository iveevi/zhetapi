#ifndef MATRIX_H_
#define MATRIX_H_

// Standard headers
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

// Library headers
#include "tensor.hpp"
#include "cuda/essentials.cuh"
#include "field.hpp"

// Redeclare minor as a matrix operation
#ifdef minor

#undef minor

#endif

namespace zhetapi {

template <class T>
class Vector;

/**
 * @brief A matrix with components of type T.
 *
 * @tparam T the type of each component.
 */
template <class T>
class Matrix : public Tensor <T>, Field <T, Matrix <T>> {
public:
	// Public aliases
	using index_type = std::pair <size_t, size_t>;

	// TODO: prune constructors
	Matrix();
	Matrix(const Matrix &);
	Matrix(const Vector <T> &);

	// Scaled
	Matrix(const Matrix &, T);

	// Reshaping a tensor
	Matrix(const Tensor <T> &, size_t, size_t);

	Matrix(size_t, size_t, T = T());

	// Lambda constructors
	Matrix(size_t, size_t, std::function <T (size_t)>);
	Matrix(size_t, size_t, std::function <T (size_t, size_t)>);

	Matrix(const std::vector <T> &);
	Matrix(const std::vector <Vector <T>> &);
	Matrix(const std::vector <std::vector <T>> &);
	Matrix(const std::initializer_list <Vector <T>> &);
	Matrix(const std::initializer_list <std::initializer_list <T>> &);

	// __cuda_dual__ Matrix(size_t, size_t, T *, bool = true);

	// Cross type operations
	template <class A>
	explicit Matrix(const Matrix <A> &);

	template <class A>
	explicit Matrix(const Tensor <A> &, size_t, size_t);

	// Methods
	inline T &get(size_t, size_t);
	inline const T &get(size_t, size_t) const;

	T norm() const;

	void resize(size_t, size_t);

	// std::get_dimensions() const;

	Matrix slice(const index_type &, const index_type &) const;

	void set(size_t, size_t, T);

	Vector <T> get_column(size_t) const;

	// Rading from a binary file (TODO: unignore later)
	void write(std::ofstream &) const;
	void read(std::ifstream &);

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
	void randomize(std::function <T ()>);

	__cuda_dual__
	void row_shur(const Vector <T> &);

	__cuda_dual__
	void stable_shur(const Matrix <T> &);

	__cuda_dual__
	void stable_shur_relaxed(const Matrix <T> &);

	// Values
	T determinant() const;

	T minor(size_t, size_t) const;
	T minor(const index_type &) const;

	T cofactor(size_t, size_t) const;
	T cofactor(const index_type &) const;

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

	std::string display() const;

	// template <class U>
	// friend std::ostream &operator<<(std::ostream &, const Matrix <U> &);

	// Special matrix generation
	static Matrix identity(size_t);

	// Miscellaneous functions
	template <class U>
	friend Vector <U> apt_and_mult(const Matrix <U> &, const Vector <U> &);

	template <class U>
	friend Vector <U> rmt_and_mult(const Matrix <U> &, const Vector <U> &);

	template <class U>
	friend Matrix <U> vvt_mult(const Vector <U> &, const Vector <U> &);

	// TODO: just use the Tensor one
	class dimension_mismatch {};
protected:
	// TODO: Looks ugly here
	T determinant(const Matrix &) const;
public:
	const Matrix &operator=(const Matrix &);

	T *operator[](size_t);
	const T *operator[](size_t) const;

	inline size_t get_rows() const;
	inline size_t get_cols() const;

	Matrix transpose() const;

	/* void operator+=(const Matrix &);
	void operator-=(const Matrix &);

	void operator*=(const T &);
	void operator/=(const T &); */

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

// Static
template <class T>
T Matrix <T> ::EPSILON = static_cast <T> (1e-10);

template <class T>
Matrix <T> ::Matrix() : Tensor <T> () {}

/**
 * @brief Component retrieval.
 *
 * @param i row index.
 * @param j column index.
 *
 * @return the component at row i and column j.
 */
template <class T>
inline T &Matrix <T> ::get(size_t i, size_t j)
{
	return this->_array[i * get_cols() + j];
}

/**
 * @brief Component retrieval.
 *
 * @param i row index.
 * @param j column index.
 *
 * @return the component at row i and column j.
 */
template <class T>
inline const T &Matrix <T> ::get(size_t i, size_t j) const
{
	return this->_array[i * get_cols() + j];
}

// TODO: make a tensor method/function for this
template <class T>
T Matrix <T> ::norm() const
{
	T sum = 0;

	for (size_t i = 0; i < this->size(); i++)
		sum += Tensor <T> ::get(i) * Tensor <T> ::get(i);

	return sqrt(sum);
}

/*
template <class T>
typename Matrix <T> ::index_type Matrix <T> ::get_dimensions() const
{
	return {get_rows(), get_cols()};
} */

template <class T>
Matrix <T> Matrix <T> ::slice(const Matrix <T> ::index_type &start, const Matrix <T> ::index_type &end) const
{
	/* The following asserts make sure the pairs are in bounds of the Matrix
	 * and that they are in order. */
	assert(start.first <= end.first && start.second <= end.second);
	assert(start.first < get_rows() && start.second < get_cols());
	assert(end.first < get_rows() && end.second < get_cols());

	/* Slicing is inclusive of the last Vector passed. */
	return Matrix <T> (
		end.first - start.first + 1,
		end.second - start.second + 1,
		[&] (size_t i, size_t j) {
			return this->_array[get_cols() * (i + start.first) + j + start.second];
		}
	);
}

/*
template <class T>
void Matrix <T> ::set(size_t row, size_t col, T val)
{
	this->_array[row][col] = val;
}

template <class T>
const T &Matrix <T> ::get(size_t row, size_t col) const
{
	return this->_array[row][col];
} */

template <class T>
Vector <T> Matrix <T> ::get_column(size_t r) const
{
	return Vector <T> (get_rows(),
		[&](size_t i) {
			return this->_array[get_cols() * i + r];
		}
	);
}

template <class T>
void Matrix <T> ::operator*=(const Matrix <T> &other)
{
	(*this) = (*this) * other;
}

// R(A) = R(A) + kR(B)
template <class T>
void Matrix <T> ::add_rows(size_t a, size_t b, T k)
{
	for (size_t i = 0; i < get_cols(); i++)
		this->_array[a][i] += k * this->_array[b][i];
}

template <class T>
void Matrix <T> ::multiply_row(size_t a, T k)
{
	for (size_t i = 0; i < get_cols(); i++)
		this->_array[a][i] *= k;
}

template <class T>
T Matrix <T> ::determinant() const
{
	return determinant(*this);
}

template <class T>
T Matrix <T> ::minor(const Matrix <T> ::index_type &pr) const
{
	Matrix <T> out(get_rows() - 1, get_cols() - 1);

	size_t a = 0;

	for (size_t i = 0; i < get_rows(); i++) {
		size_t b = 0;
		if (i == pr.first)
			continue;

		for (size_t j = 0; j < get_cols(); j++) {
			if (j == pr.second)
				continue;

			out[a][b++] = this->_array[i * get_cols() + j];
		}

		a++;
	}

	return determinant(out);
}

template <class T>
T Matrix <T> ::minor(size_t i, size_t j) const
{
	return minor({i, j});
}

template <class T>
T Matrix <T> ::cofactor(const Matrix <T> ::index_type &pr) const
{
	return (((pr.first + pr.second) % 2) ? -1 : 1) * minor(pr);
}

template <class T>
T Matrix <T> ::cofactor(size_t i, size_t j) const
{
	return cofactor({i, j});
}

template <class T>
Matrix <T> Matrix <T> ::inverse() const
{
	return adjugate() / determinant();
}

template <class T>
Matrix <T> Matrix <T> ::adjugate() const
{
	return cofactor().transpose();
}

template <class T>
Matrix <T> Matrix <T> ::cofactor() const
{
	return Matrix(get_rows(), get_cols(),
		[&](size_t i, size_t j) {
			return cofactor(i, j);
		}
	);
}

// TODO: put these property checks in another file

/**
 * @brief Checks whether the matrix is symmetric.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is symmetric and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_symmetric(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = i + 1; j < c; j++) {
			// Avoid using abs
			T d = get(i, j) - get(j, i);
			if (d > epsilon || d < -epsilon)
				return false;
		}
	}

	return true;
}

/**
 * @brief Checks whether the matrix is diagonal.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is diagonal and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_diagonal(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if (i == j)
				continue;

			if (get(i, j) < -epsilon || get(i, j) > epsilon)
				return false;
		}
	}

	return true;
}

/**
 * @brief Checks whether the matrix is identity matrix (for any dimension).
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is the identity and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_identity(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	if (r != c)
		return false;

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if (i == j) {
				T d = 1 - get(i, j);
				if (d < -epsilon || d > epsilon)
					return false;
			} else if (get(i, j) < -epsilon || get(i, j) > epsilon) {
				return false;
			}
		}
	}

	return true;
}

/**
 * @brief Checks whether the matrix is orthogonal.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is orthogonal and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_orthogonal(const T &epsilon) const
{
	return is_identity(*this * transpose());
}

/**
 * @brief Checks whether the matrix is lower triangular.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is lower triangular and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_lower_triangular(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if ((i > j) && get(i, j) > epsilon)
				return false;
		}
	}

	return true;
}

/**
 * @brief Checks whether the matrix is upper triangular.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is upper triangular and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_upper_triangular(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if ((i < j) && get(i, j) > epsilon)
				return false;
		}
	}

	return true;
}

template <class T>
Matrix <T> Matrix <T> ::identity(size_t dim)
{
	return Matrix(dim, dim,
		[](size_t i, size_t j) {
			return (i == j) ? 1 : 0;
		}
	);
}

// Private helper methods
template <class T>
T Matrix <T> ::determinant(const Matrix <T> &a) const
{
	/* The determinant of an abitrary Matrix is defined only if it is a
	 * square Matrix.
	 */
	assert((a.get_rows() == a.get_cols()) && (a.get_rows() > 0));

	size_t n;
	size_t t;

	n = a.get_rows();

	if (n == 1)
		return a[0][0];
	if (n == 2)
		return a[0][0] * a[1][1] - a[1][0] * a[0][1];

	T det = 0;

	for (size_t i = 0; i < n; i++) {
		Matrix <T> temp(n - 1, n - 1);

		for (size_t j = 0; j < n - 1; j++) {
			t = 0;

			for (size_t k = 0; k < n; k++) {
				if (k == i)
					continue;
				temp[j][t++] = a[j + 1][k];
			}
		}

		det += ((i % 2) ? -1 : 1) * a[0][i] * determinant(temp);
	}

	return det;
}

// TODO: perform straight copy...
template <class T>
Matrix <T> ::Matrix(const Matrix <T> &other)
		: Tensor <T> (other.get_rows(), other.get_cols())
{
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] = other._array[i];
}

template <class T>
template <class A>
Matrix <T> ::Matrix(const Matrix <A> &other)
		: Tensor <T> (other.get_rows(), other.get_cols())
{
	const A *array = other[0];
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] = array[i];
}

// TODO: Do all initialization inline or use Tensor copy constructor
template <class T>
Matrix <T> ::Matrix(const Vector <T> &other)
		: Tensor <T> (other)
{
	this->reshape({other.size(), 1});
}

template <class T>
Matrix <T> ::Matrix(const Matrix <T> &other, T k)
{
	if (this != &other) {
		// Use a macro
		this->_array = new T[other._size];
		this->get_rows() = other.get_rows();
		this->get_cols() = other.get_cols();

		this->size() = other._size;
		for (size_t i = 0; i < this->size(); i++)
			this->_array[i] = k * other._array[i];

		this->_dims = 2;
		this->_dim = new size_t[2];

		this->_dim[0] = this->get_rows();
		this->_dim[1] = this->get_cols();
	}
}

// Tensor reshaper
// TODO: should not copy the data (or create new)
template <class T>
Matrix <T> ::Matrix(const Tensor <T> &other, size_t rows, size_t cols)
		: Tensor <T> (rows, cols)
{
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] = other.get(i);
}

template <class T>
template <class A>
Matrix <T> ::Matrix(const Tensor <A> &other, size_t rows, size_t cols)
		: Tensor <T> (rows, cols)
{
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] = static_cast <T> (other.get(i));
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, T val)
		: Tensor <T> (rs, cs)
{
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] = val;
}

template <class T>
const Matrix <T> &Matrix <T> ::operator=(const Matrix <T> &other)
{
	if (this != &other)
		Tensor <T> ::operator=(other);

	return *this;
}

template <class T>
void Matrix <T> ::resize(size_t rs, size_t cs)
{
	if (rs != get_rows() || cs != get_cols()) {
		this->size() = rs * cs;

		this->_clear();

		this->_array = new T[this->size()];

		if (!this->_dim) {
			this->_dims = 2;
			this->_dim = new size_t[2];

			this->_dim[0] = this->get_rows();
			this->_dim[1] = this->get_cols();
		}
	}
}

template <class T>
T *Matrix <T> ::operator[](size_t i)
{
	return (this->_array.get() + i * get_cols());
}

template <class T>
const T *Matrix <T> ::operator[](size_t i) const
{
	return (this->_array.get() + i * get_cols());
}

template <class T>
inline size_t Matrix <T> ::get_rows() const
{
	return this->dimension(0);
}

template <class T>
inline size_t Matrix <T> ::get_cols() const
{
	// TODO: do we absolutely need this check?
	// do we need to check for rows?
	// or do we assume that every matrix (including vectors)
	// have at least a row
	if (this->dimensions() < 2)
		return 1;

	return this->dimension(1);
}

template <class T>
Matrix <T> Matrix <T> ::transpose() const
{
	// TODO: surely something more efficient
	return Matrix(get_cols(), get_rows(),
		[&](size_t i, size_t j) {
			return this->_array[j * get_cols() + i];
		}
	);
}

template <class T>
void Matrix <T> ::row_shur(const Vector <T> &other)
{
	// Not strict (yet)
	for (size_t i = 0; i < get_rows(); i++) {
		T *arr = &(this->_array[i * get_cols()]);

		for (size_t j = 0; j < get_cols(); j++)
			arr[j] *= other._array[i];
	}
}

template <class T>
void Matrix <T> ::stable_shur(const Matrix <T> &other)
{
	// TODO: add a same_shape function
	if (!((other.safe_dim_size(0) == this->safe_dim_size(0))
		&& (other.safe_dim_size(1) == this->safe_dim_size(1))))
		throw typename Matrix <T> ::dimension_mismatch();

	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] *= other._array[i];
}

template <class T>
void Matrix <T> ::stable_shur_relaxed(const Matrix <T> &other)
{
	// Loop for the limits of the other
	for (size_t i = 0; i < other._size; i++)
		this->_array[i] *= other._array[i];
}

/* template <class T>
void Matrix <T> ::operator+=(const Matrix <T> &other)
{
	assert(get_rows() == other.get_rows() && get_cols() == other.get_cols());

	for (size_t i = 0; i < get_rows(); i++) {
		for (size_t j = 0; j < get_cols(); j++)
			this->_array[i * get_cols() + j] += other._array[i * get_cols() + j];
	}
}

template <class T>
void Matrix <T> ::operator-=(const Matrix <T> &other)
{
	assert(get_rows() == other.get_rows() && get_cols() == other.get_cols());

	for (size_t i = 0; i < get_rows(); i++) {
		for (size_t j = 0; j < get_cols(); j++)
			this->_array[i * get_cols() + j] -= other._array[i * get_cols() + j];
	}
}

// TODO: Remove as it is done in tensor already
template <class T>
void Matrix <T> ::operator*=(const T &x)
{
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] *= x;
}

template <class T>
void Matrix <T> ::operator/=(const T &x)
{
	for (size_t i = 0; i < this->size(); i++)
		this->_array[i] /= x;
} */

// TODO: Remove as it is done in tensor already
template <class T>
Matrix <T> operator+(const Matrix <T> &a, const Matrix <T> &b)
{
	Matrix <T> c = a;
	c += b;
	return c;
}

template <class T>
Matrix <T> operator-(const Matrix <T> &a, const Matrix <T> &b)
{
	Matrix <T> c = a;
	c -= b;
	return c;
}

template <class T, class U>
Matrix <T> operator*(const Matrix <T> &A, const Matrix <U> &B)
{
	if (A.get_cols() != B.get_rows()) {
		std::cout << "A.get_rows() = " << A.get_rows() << std::endl;
		std::cout << "A.get_cols() = " << A.get_cols() << std::endl;
		std::cout << "B.get_rows() = " << B.get_rows() << std::endl;
		std::cout << "B.get_cols() = " << B.get_cols() << std::endl;
		throw typename Matrix <T> ::dimension_mismatch();
	}

	size_t rs = A.get_rows();
	size_t cs = B.get_cols();

	size_t kmax = B.get_rows();

	Matrix <T> C(rs, cs);

	// TODO: Use a parallel for
	for (size_t i = 0; i < rs; i++) {
		const T *Ar = A[i];
		T *Cr = C[i];

		for (size_t k = 0; k < kmax; k++) {
			const U *Br = B[k];

			T a = Ar[k];
			for (size_t j = 0; j < cs; j++)
				Cr[j] += T(a * Br[j]);
		}
	}

	return C;
}

template <class T>
Matrix <T> operator*(const Matrix <T> &a, const T &scalar)
{
	return Matrix <T> (a.get_rows(), a.get_cols(),
		[&](size_t i, size_t j) {
			return a[i][j] * scalar;
		}
	);
}

template <class T>
Matrix <T> operator*(const T &scalar, const Matrix <T> &a)
{
	return a * scalar;
}

template <class T>
Matrix <T> operator/(const Matrix <T> &a, const T &scalar)
{
	return Matrix <T> (a.get_rows(), a.get_cols(),
		[&](size_t i, size_t j) {
			return a[i][j] / scalar;
		}
	);
}

template <class T>
Matrix <T> operator/(const T &scalar, const Matrix <T> &a)
{
	return a / scalar;
}

template <class T>
bool operator==(const Matrix <T> &a, const Matrix <T> &b)
{
	if (a.get_dimensions() != b.get_dimensions())
		return false;

	for (size_t i = 0; i  < a.get_rows(); i++) {
		for (size_t j = 0; j < a.get_cols(); j++) {
			if (a[i][j] != b[i][j])
				return false;
		}
	}

	return true;
}

template <class T>
Matrix <T> shur(const Matrix <T> &a, const Matrix <T> &b)
{
	if (!((a.get_rows() == b.get_rows())
		&& (a.get_cols() == b.get_cols())))
		throw typename Matrix <T> ::dimension_mismatch();

	return Matrix <T> (a.get_rows(), b.get_cols(),
		[&](size_t i, size_t j) {
			return a[i][j] * b[i][j];
		}
	);
}

template <class T>
Matrix <T> inv_shur(const Matrix <T> &a, const Matrix <T> &b)
{
	if (!((a.get_rows() == b.get_rows())
		&& (a.get_cols() == b.get_cols())))
		throw typename Matrix <T> ::dimension_mismatch();

	return Matrix <T> (a.get_rows(), b.get_cols(),
		[&](size_t i, size_t j) {
			return a[i][j] / b[i][j];
		}
	);
}

// Computes A * B + C
template <class T, class U, class V>
Matrix <T> fma(const Matrix <T> &A, const Matrix <U> &B, const Matrix <V> &C)
{
	if (A.get_cols() != B.get_rows())
		throw typename Matrix <T> ::dimension_mismatch();

	if (A.get_rows() != C.get_rows() || B.get_cols() != C.get_cols())
		throw typename Matrix <T> ::dimension_mismatch();

	size_t rs = A.get_rows();
	size_t cs = B.get_cols();

	size_t kmax = B.get_rows();

	Matrix <T> D = C;

	for (size_t i = 0; i < rs; i++) {
		const T *Ar = A[i];
		T *Dr = D[i];

		for (size_t k = 0; k < kmax; k++) {
			const U *Br = B[k];

			T a = Ar[k];
			for (size_t j = 0; j < cs; j++)
				Dr[j] += T(a * Br[j]);
		}
	}

	return D;
}

// Computes ka * A * B + kb * C
template <class T, class U, class V>
Matrix <T> fmak(const Matrix <T> &A, const Matrix <U> &B, const Matrix <V> &C, T ka, T kb)
{
	if (A.get_cols() != B.get_rows())
		throw typename Matrix <T> ::dimension_mismatch();

	if (A.get_rows() != C.get_rows() || B.get_cols() != C.get_cols())
		throw typename Matrix <T> ::dimension_mismatch();

	size_t rs = A.get_rows();
	size_t cs = B.get_cols();

	size_t kmax = B.get_rows();

	Matrix <T> D(C, kb);

	for (size_t i = 0; i < rs; i++) {
		const T *Ar = A[i];
		T *Dr = D[i];

		for (size_t k = 0; k < kmax; k++) {
			const U *Br = B[k];

			T a = Ar[k] * ka;
			for (size_t j = 0; j < cs; j++)
				Dr[j] += T(a * Br[j]);
		}
	}

	return D;
}

#ifdef __AVR

template <class T>
String Matrix <T> ::display() const
{
	String out = "[";

	for (size_t i = 0; i < get_rows(); i++) {
		if (get_cols() > 1) {
			out += '[';

			for (size_t j = 0; j < get_cols(); j++) {
				out += String(this->_array[i * get_cols() + j]);

				if (j != get_cols() - 1)
					out += ", ";
			}

			out += ']';
		} else {
			out += String(this->_array[i * get_cols()]);
		}

		if (i < get_rows() - 1)
			out += ", ";
	}

	return out + "]";
}

#endif

/* Externally defined methods
template <class T>
Matrix <T> Tensor <T> ::cast_to_matrix(size_t r, size_t c) const
{
	// Return a slice-vector
	return Matrix <T> (r, c, _array);
} */

}

// TODO use _CUDACC_ instead of _zhp_cuda and make _cuda files
// #include "primitives/matrix_prims.hpp"

#ifndef __AVR

#include "matrix_cpu.hpp"

#endif

#endif
