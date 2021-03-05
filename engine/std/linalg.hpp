#ifndef LINALG_H_
#define LINALG_H_

// C/C++ headers
#include <utility>

// Engine headers
#include <matrix.hpp>
#include <vector.hpp>

namespace zhetapi {

namespace linalg {

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

}

}

#endif
