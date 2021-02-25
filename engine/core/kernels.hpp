#ifndef KERNELS_H_
#define KERNELS_H_

// Engine headers
#include <matrix.hpp>
#include <vector.hpp>

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
	size_t rs = M.__rows;
	size_t cs = M.__cols;

	Vector <T> out(rs, T(0));

	size_t k = V.__size;
	for (size_t i = 0; i < rs; i++) {
		T acc = M.__array[i * cs];

		for (size_t j = 0; j < k; j++)
			acc += M.__array[i * cs + 1 + j] * V.__array[j];

		out.__array[i] = acc;
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
	size_t rs = M.__rows;
	size_t cs = M.__cols;

	Vector <T> out(cs - 1, T(0));
	for (size_t i = 1; i < cs; i++) {
		T acc = 0;

		for (size_t k = 0; k < rs; k++)
			acc += M.__array[k * cs + i] * V.__array[k];

		out.__array[i - 1] = acc;
	}

	return out;
}

template <class T>
void rmt_and_mult_ref(const Matrix <T> &M, Vector <T> &V)
{
	size_t rs = M.__rows;
	size_t cs = M.__cols;

	T *tmp = new T[cs - 1];
	for (size_t i = 1; i < cs; i++) {
		T acc = 0;

		for (size_t k = 0; k < rs; k++)
			acc += M.__array[k * cs + i] * V.__array[k];

		tmp[i - 1] = acc;
	}

	memcpy(V.__array, tmp, sizeof(T) * (cs - 1));

	delete[] tmp;
}

/**
 * Computes V * (Vt)^T (transpose). Speed-up comes from the fact that we avoid
 * creating the transpose vector.
 */
template <class T>
Matrix <T> vvt_mult(const Vector <T> &V, const Vector <T> &Vt)
{
	size_t rs = V.__size;
	size_t cs = Vt.__size;

	Matrix <T> out(rs, cs, T(0));

	size_t n = rs * cs;
	for (size_t i = 0; i < n; i++)
		out.__array[i] = V.__array[i / cs] * Vt.__array[i % cs];

	return out;
}

template <class T>
void vvt_mult_ref(Matrix <T> &A, const Vector <T> &V, const Vector <T> &Vt)
{
	size_t rs = V.__size;
	size_t cs = Vt.__size;

	A.resize(rs, cs);

	size_t n = rs * cs;
	for (size_t i = 0; i < n; i++)
		A.__array[i] = V.__array[i / cs] * Vt.__array[i % cs];
}

}

#endif
