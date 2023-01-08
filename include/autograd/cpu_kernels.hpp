#ifndef ZHETAPI_AUTOGRAD_CPU_KERNELS_H_
#define ZHETAPI_AUTOGRAD_CPU_KERNELS_H_

// Standard headers
#include <cstdlib>

namespace zhetapi {

namespace detail {

namespace autograd {

// TODO: put in source file
inline void fma_matrix_vector(float *out, const float *matrix, const float *bias, const float *input, size_t rows, size_t cols)
{
#pragma omp parallel for
	for (size_t i = 0; i < rows; i++) {
		float sum = 0;

		for (size_t j = 0; j < cols; j++)
			sum += matrix[i * cols + j] * input[j];

		out[i] = sum + bias[i];
	}
}

inline void mul_vector_vector_transpose(float *out, const float *a, const float *b, size_t na, size_t nb)
{
#pragma omp parallel for
	for (size_t i = 0; i < na; i++) {
		for (size_t j = 0; j < nb; j++)
			out[i * nb + j] = a[i] * b[j];
	}
}

inline void mul_matrix_transpose_vector(float *out, const float *matrix, const float *vector, size_t na, size_t nb)
{
#pragma omp parallel for
	for (size_t i = 0; i < na; i++) {
		float sum = 0;
		
		for (size_t j = 0; j < nb; j++)
			sum += matrix[i + j * na] * vector[j];

		out[i] = sum;
	}
}

}

}

}

#endif
