#ifndef LINALG_H_
#define LINALG_H_

// C/C++ headers
#include <utility>
#include <iomanip>

// Engine headers
#include <matrix.hpp>
#include <vector.hpp>

#include <core/common.hpp>

namespace zhetapi {

namespace linalg {

typedef Vector <long double> Vec;
typedef Matrix <long double> Mat;

extern const long double GAMMA;
extern const long double EPSILON;

// MatrixFactorization class
template <class T, size_t N>
class MatrixFactorization {
protected:
	Matrix <T>	_terms[N];
public:
	MatrixFactorization(const std::vector <Matrix <T>> &terms) {
		for (size_t i = 0; i < N; i++)
			_terms[i] = terms[i];
	}
	
	// Variadic constructor
	template <class ... U>
	MatrixFactorization(const Matrix <T> &A, U ... args) {
		std::vector <Matrix <T>> terms {A};

		collect(terms, args...);

		// Skip overhead of std::vector
		// by adding a collect function
		// for pointer arrays
		for (size_t i = 0; i < N; i++)
			_terms[i] = terms[i];
	}

	Matrix <T> product() const {
		// Check for appropriate number
		if (N <= 0)
			return Matrix <T> ::identity(1);
		
		Matrix <T> prod = _terms[0];

		for (size_t i = 1; i < N; i++)
			prod *= _terms[i];
		
		return prod;
	}
};

// Separate matrix/vector methods from algorithms
template <class T>
std::ostream &pretty(std::ostream &os, const Matrix <T> &mat)
{
	size_t r = mat.get_rows();
	size_t c = mat.get_cols();

	for (size_t i = 0; i < r; i++) {
		os << "|";

		for (size_t j = 0; j < c; j++) {
			os << std::setw(8) << std::fixed
				<< std::setprecision(4) << mat[i][j];

			if (j < c - 1)
				os << "\t";
		}

		os << "|";
		if (i < r - 1)
			os << "\n";
	}

	return os;
}

// Checks
template <class T>
bool is_diagonal(const Matrix <T> &mat, const T &epsilon = EPSILON)
{
	size_t r = mat.get_rows();
	size_t c = mat.get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if ((i != j) && mat[i][j] > epsilon)
				return false;
		}
	}

	return true;
}

template <class T>
bool is_identity(const Matrix <T> &mat, const T &epsilon = EPSILON)
{
	size_t r = mat.get_rows();
	size_t c = mat.get_cols();

	if (r != c)
		return false;

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if (i == j) {
				if (fabs(1 - mat[i][j]) > epsilon)
					return false;
			} else if (mat[i][j] > epsilon) {
				return false;
			}
		}
	}

	return true;
}

template <class T>
bool is_orthogonal(const Matrix <T> &mat, const T &epsilon = EPSILON)
{
	return is_identity(mat * mat.transpose());
}

template <class T>
bool is_lower_triangular(const Matrix <T> &mat, const T &epsilon = EPSILON)
{
	size_t r = mat.get_rows();
	size_t c = mat.get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if ((i > j) && mat[i][j] > epsilon)
				return false;
		}
	}

	return true;
}

template <class T>
bool is_right_triangular(const Matrix <T> &mat, const T &epsilon = EPSILON)
{
	size_t r = mat.get_rows();
	size_t c = mat.get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if ((i < j) && mat[i][j] > epsilon)
				return false;
		}
	}

	return true;
}

// Reshaping functions
template <class T>
Vector <T> flatten(const Matrix <T> &mat)
{
	size_t r = mat.get_rows();
	size_t c = mat.get_cols();

	return Vector <T> (r * c,
		[&](size_t i) {
			return mat[i / c][i % c];
		}
	);
}

template <class T>
Matrix <T> fold(const Vector <T> &vec, size_t r, size_t c)
{
	return Matrix <T> (r, c,
		[&](size_t i, size_t j) {
			return vec[i * c + j];
		}
	);
}

template <class T>
Tensor <T> reshape(const Vector <T> &);

// Create a diagonal matrix
template <class T>
Matrix <T> diag(const std::vector <T> &cs)
{
	return Matrix <T> (cs.size(), cs.size(),
		[&](size_t i, size_t j) {
			return (i == j) ? cs[i] : 0;
		}
	);
}

template <class T, class ... U>
Matrix <T> diag(T x, U ... args)
{
	std::vector <T> bin {x};

	collect(bin, args...);

	return diag(bin);
}

/**
 * @brief Projects one vector onto another.
 *
 * @param u The base of the projection.
 * @param v The vector to be projected.
 *
 * @return The value of proj_u(v), the projection of v onto u.
 */
template <class T>
Vector <T> proj(const Vector <T> &u, const Vector <T> &v)
{
	return (inner(u, v) / inner(u, u)) * u;
}

// QR factorization class
template <class T>
class QR : public MatrixFactorization <T, 2> {
public:
	QR(const Matrix <T> &Q, const Matrix <T> &R)
			: MatrixFactorization <T, 2> (Q, R) {}
	
	Matrix <T> q() const {
		return this->_terms[0];
	}

	Matrix <T> r() const {
		return this->_terms[1];
	}
};

// LQ factorization class
template <class T>
class LQ : public MatrixFactorization <T, 2> {
public:
	LQ(const Matrix <T> &L, const Matrix <T> &Q)
			: MatrixFactorization <T, 2> (L, Q) {}
	
	Matrix <T> l() const {
		return this->_terms[0];
	}

	Matrix <T> q() const {
		return this->_terms[1];
	}
};

/**
 * @brief Performs QR decomposition, where Q is an orthogonal matrix and R is
 * an upper triangular matrix.
 *
 * @param A The matrix to be decomposed.
 *
 * @return A QR factorization object containing the matrices Q and R.
 */
template <class T>
QR <T> qr_decompose(const Matrix <T> &A)
{
	// Assume that A is square for now
	//
	// TODO: Add a method .is_square()
	assert(A.get_rows() == A.get_cols());

	// Get dimension
	size_t n = A.get_rows();

	std::vector <Vector <T>> us;
	std::vector <Vector <T>> es;

	Vector <T> u = A.get_column(0);

	// TODO: Should optimize this algorithm (using indices)

	us.push_back(u);
	es.push_back(u.normalized());
	
	for (size_t i = 1; i < n; i++) {
		Vector <T> ai = A.get_column(i);

		u = ai;
		for (size_t k = 0; k < i; k++)
			u -= proj(us[k], ai);
		
		us.push_back(u);
		es.push_back(u.normalized());
	}
	
	Matrix <T> Q(es);

	Matrix <T> R(n, n);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = i; j < n; j++)
			R[i][j] = inner(es[i], A.get_column(j));
	}

	// Make a structure for matrix
	// factorization of a general form
	// 
	// A * B * C * ...
	return QR <T> (Q, R);
}

/**
 * @brief Performs LQ factorization, where L is a lower triangular matrix and Q
 * is an orthogonal matrix/
 *
 * @param A The matrix to be factorized.
 *
 * @return A pair containing the matrices L and Q, in that order
 */
template <class T>
LQ <T> lq_decompose(const Matrix <T> &A)
{
	// Use a more verbose method for better
	// accuracy and efficiency
	auto qr = qr_decompose(A);

	return LQ <T> (qr.q().transpose(), qr.r().transpose());
}

/**
 * @brief Performs the QR algorithm to compute the eigenvalues of a square
 * matrix.
 *
 * @param A The matrix whos eigenvalues are to be computed.
 * @param epsilon The precision threshold; when all values not in the diagonal
 * of the matrix are less than or equal to epsilon, the algorithm terminates.
 * @param limit The maximum number of iterations to perform.
 *
 * @return A vector of the diagonal elements of the matrix after termination. If
 * the algorithm is successful, this should contain the (real) eigenvalues of A.
 */
template <class T>
Vector <T> qr_algorithm(
		const Matrix <T> &A,
		T epsilon = T(1e-10),
		size_t limit = 1000)
{
	// Assume that A is square for now
	size_t n = A.get_rows();

	Matrix <T> U = A;
	for (size_t i = 0; i < limit; i++) {
		auto qr = qr_decompose(U);

		U = qr.r() * qr.q();

		bool terminate = true;
		for (size_t i = 0; i < n * n; i++) {
			if (*(U[i]) > epsilon)
				terminate = false;
		}

		if (terminate)
			break;
	}

	Vector <T> eigenvalues(n);
	for (size_t i = 0; i < n; i++)
		eigenvalues[i] = U[i][i];

	return eigenvalues;
}

Vec pslq(const Vec &, long double = GAMMA, long double = EPSILON);

template <class T>
Matrix <T> exp(const Matrix <T> &A, size_t pow)
{
	// Base cases
	if (pow == 0)
		return Matrix <T> ::indentity(A.get_rows());
	
	if (pow == 1)
		return A;
	
	// Use recusion
	if (pow % 2)
		return A * pow(A, (pow - 1)/2);
	
	return pow(A, pow/2);
}

}

}

#endif
