#ifndef KERNELS_H_
#define KERNELS_H_

// Engine headers
#include "matrix.hpp"
#include "vector.hpp"

/**
 * This file contains CPU "kernels" which speed up computation in other parts of
 * the library, such as Neural Network training. Some of these kernels may be
 * moved to become part of the public API.
 */
namespace zhetapi {

/**
 * Computes M * V', where V' is V with 1 appended to the top. Speed-up is due to
 * the fact that a new vector is not being created and copied.
 */
template <class T>
Vector <T> apt_and_mult(const Matrix <T> &M, const Vector <T> &V)
{
	size_t rs = M._rows;
	size_t cs = M._cols;

	Vector <T> out(rs, T(0));

	size_t k = V._size;
	for (size_t i = 0; i < rs; i++) {
		T acc = M._array[i * cs];

		for (size_t j = 0; j < k; j++)
			acc += M._array[i * cs + 1 + j] * V._array[j];

		out._array[i] = acc;
	}

	return out;
}

/**
 * Computes U', where U = M * V and U' is U without the first element. Speed-up
 * is again due to the fact that a new vector is not being created.
 */
template <class T>
Vector <T> rmt_and_mult(const Matrix <T> &M, const Vector <T> &V)
{
	size_t rs = M._rows;
	size_t cs = M._cols;

	Vector <T> out(cs - 1, T(0));
	/* for (size_t i = 1; i < cs; i++) {
		T acc = 0;

		for (size_t k = 0; k < rs; k++)
			acc += M._array[k * cs + i] * V._array[k];

		out._array[i - 1] = acc;
	} */

	// Reverse loops
	for (size_t k = 0; k < rs; k++) {
		const T *arr = &(M._array[k * cs]);
		T v = V._array[k];

		for (size_t i = 1; i < cs; i++)
			out._array[i - 1] = arr[i] * v;
	}

	return out;
}

/**
 * Computes V * (Vt)^T (transpose). Speed-up comes from the fact that we avoid
 * creating the transpose vector.
 */
template <class T>
Matrix <T> vvt_mult(const Vector <T> &V, const Vector <T> &Vt)
{
	size_t rs = V._size;
	size_t cs = Vt._size;
	
	size_t n = rs * cs;

	T *tmp = new T[n];
	for (size_t i = 0; i < n; i++)
		tmp[i] = V._array[i / cs] * Vt._array[i % cs];

	return Matrix <T> (rs, cs, tmp, false);
}

}

#endif
