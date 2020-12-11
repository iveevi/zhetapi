#ifndef MATRIX_CUH_
#define MATRIX_CUH_

#define ZHP_CUDA

#include <matrix.hpp>

namespace zhetapi {

	// Matrix member operators
        template <class T>
	__host__ __device__
        void Matrix <T> ::operator+=(const Matrix <T> &other)
        {
                assert(__rows == other.__rows && __cols == other.__cols);

                for (size_t i = 0; i < __rows; i++) {
                        for (size_t j = 0; j < __cols; j++)
                                this->__array[i * __cols + j] += other.__array[i * __cols + j];
                }
        }

        template <class T>
	__host__ __device__
        void Matrix <T> ::operator-=(const Matrix <T> &other)
        {
                assert(__rows == other.__rows && __cols == other.__cols);

                for (size_t i = 0; i < __rows; i++) {
                        for (size_t j = 0; j < __cols; j++)
                                this->__array[i * __cols + j] -= other.__array[i * __cols + j];
                }
        }

	// Matrix non-member operators
        template <class T>
	__host__ __device__
        Matrix <T> operator+(const Matrix <T> &a, const Matrix <T> &b)
        {
                assert(a.__rows == b.__rows && a.__cols == b.__cols);
                return Matrix <T> (a.__rows, a.__cols,
			[&](size_t i, size_t j) {
                        	return a[i][j] + b[i][j];
                	}
		);
        }

        template <class T>
	__host__ __device__
        Matrix <T> operator-(const Matrix <T> &a, const Matrix <T> &b)
        {
                assert(a.__rows == b.__rows && a.__cols == b.__cols);
		return Matrix <T> (a.__rows, a.__cols,
			[&](size_t i, size_t j) {
                        	return a[i][j] - b[i][j];
			}
		);
        }

}

#endif
