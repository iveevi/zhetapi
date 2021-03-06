#ifndef LINALG_H_
#define LINALG_H_

// C/C++ headers
#include <utility>

// Engine headers
#include <matrix.hpp>
#include <vector.hpp>

#include <core/common.hpp>

namespace zhetapi {

namespace linalg {

template <class T>
Matrix <T> diag(const std::vector <T> &cs)
{
	return Matrix <T> (cs.size(), cs.size(),
		[&](size_t i, size_t j) {
			return (i == j) ? cs[i] : 0;
		}
	);
}

// TODO: Fix bug with this one
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

/**
 * @brief Performs QR decomposition, where Q is an orthogonal matrix and R is
 * an upper triangular matrix.
 *
 * @param A The matrix to be decomposed.
 *
 * @return A pair containing the matrices Q and R.
 */
template <class T>
std::pair <Matrix <T>, Matrix <T>> qr_decompose(const Matrix <T> &A)
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

	return {Q, R};
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

		U = qr.second * qr.first;

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

Vector <long long int> pslq(const Vector <long double> &);

}

}

#endif
