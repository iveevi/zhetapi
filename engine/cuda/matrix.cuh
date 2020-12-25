#ifndef MATRIX_CUH_
#define MATRIX_CUH_

#define ZHP_CUDA

// CUDA headers
#include <cuda/tensor.cuh>

// Engine headers
#include <matrix.hpp>

namespace zhetapi {

        template <class T>
	__host__ __device__
        Matrix <T> ::Matrix() : __rows(0), __cols(0), Tensor <T> () {}
        
	template <class T>
	__host__ __device__
        Matrix <T> ::Matrix(bool dev) : __rows(0), __cols(0), Tensor <T> (dev) {}

        template <class T>
	__host__ __device__
        Matrix <T> ::Matrix(const Matrix <T> &other) : __rows(other.__rows), __cols(other.__cols), Tensor <T>
                                                       (other.__rows, other.__cols, T())
        {
                for (int i = 0; i < __rows; i++) {
                        for (int j = 0; j < __cols; j++)
                                this->__array[__cols * i + j] = other.__array[__cols * i + j];
                }
        }

	template <class T>
	__host__ __device__
        Matrix <T> ::Matrix(const Vector <T> &other) : __rows(other.__rows), __cols(1), Tensor <T>
                                                       (other.__rows, 1, T())
        {
                for (int i = 0; i < __rows; i++)
		        this->__array[i] = other.__array[i];
        }

        template <class T>
	__host__ __device__
        Matrix <T> ::Matrix(size_t rs, size_t cs, T val) : Tensor <T> (rs, cs, T())
        {
                __rows = rs;
                __cols = cs;
                
                for (int i = 0; i < __rows; i++) {
                        for (int j = 0; j < __cols; j++)
                                this->__array[__cols * i + j] = val;
                }
        }

        template <class T>
	__host__ __device__
        Matrix <T> ::Matrix(size_t rs, size_t cs, T *arr) : Tensor <T> (rs, cs, T())
        {
                __rows = rs;
                __cols = cs;

                for (int i = 0; i < this->__size; i++)
			this->__array[i] = arr[i];
        }

        template <class T>
	template <class F>
	__host__ __device__
        Matrix <T> ::Matrix(size_t rs, size_t cs, F gen) : Tensor <T> (rs, cs, T())
        {
                __rows = rs;
                __cols = cs;
                
                for (int i = 0; i < __rows; i++) {
                        for (int j = 0; j < __cols; j++)
                                this->__array[__cols * i + j] = gen(i, j);
                }
        }
	
	template <class T>
	__host__ __device__
	const Matrix <T> &Matrix <T> ::operator=(const Matrix <T> &other)
	{
		if (this != &other) {
			delete[] this->__array;
			delete[] this->__dim;

			this->__array = new T[other.__size];
			this->__rows = other.__rows;
			this->__cols = other.__cols;

			this->__size = other.__size;
			for (size_t i = 0; i < this->__size; i++)
				this->__array[i] = other.__array[i];
			
			this->__dims = 2;
			this->__dim = new size_t[2];

			this->__dim[0] = this->__rows;
			this->__dim[1] = this->__cols;
		}

		return *this;
	}

	// TODO: Turn this method into a virtual method with the base as Tensor
	template <class T>
	void Matrix <T> ::copy_to_device(Matrix <T> other)
	{
		this->__on_device = true;

		if (this->__array)
			delete[] this->__array;

		if (this->__dim)
			delete[] this->__dim;

		cudaMalloc(&this->__array, sizeof(T) * other.__size);
		cudaMemcpy(this->__array, other.__array, sizeof(T) *
				other.__size, cudaMemcpyHostToDevice);

		__rows = other.__rows;
		__cols = other.__cols;

		this->__size = __rows * __cols;

		this->__dims = 2;

		cudaMalloc(&this->__dim, 2 * sizeof(size_t));
		cudaMemcpy(this->__dim, other.__dim, 2 * sizeof(size_t),
				cudaMemcpyHostToDevice);
	}

	// TODO: Turn this method into a virtual method with the base as Tensor
	template <class T>
	void Matrix <T> ::transfer_from_device(Matrix <T> &other)
	{
		if (other.__array)
			delete[] other.__array;

		if (other.__dim)
			delete[] other.__dim;

		other.__rows = __cols;
		other.__cols = __rows;

		other.__size = __rows * __cols;
		
		other.__array = new T[this->__size];

		cudaMemcpy(other.__array, this->__array, sizeof(T) *
				this->__size, cudaMemcpyDeviceToHost);
		cudaCheckError(this->__array);

		other.__dims = 2;

		other.__dim = new size_t[2];
		other.__dim[0] = __rows;
		other.__dim[1] = __cols;
	}
	
	// TODO: Turn this method into a virtual method with the base as Tensor
	template <class T>
	__host__ __device__
	void Matrix <T> ::stable_transfer(const Matrix <T> &other)
	{
		// TODO: Add error handling for when the dimensions mismatch
		for (int i = 0; i < this->__size; i++)
			this->__array[i] = other.__array[i];
	}
	
	template <class T>
	__host__ __device__
	void Matrix <T> ::show(int tid) const
	{
		printf("#ID = %d, ", tid);
		printf("array = %p (", this->__array);

		for (size_t i = 0; i < this->__size; i++) {
			printf("%f", this->__array[i]);

			if (i < this->__size - 1)
				printf(", ");
		}

		printf(") dim = %p ", this->__dim);
		printf("(%d), ", this->__dim[0]);
		printf("rows = %d, ", __rows);
		printf("cols = %d, ", __cols);
		printf("size = %d, ", this->__size);
		printf("dims = %d\n", this->__dims);
	}
        
	template <class T>
	__host__ __device__
        T *Matrix <T> ::operator[](size_t i)
        {
                return (this->__array + i * __cols);
        }

        template <class T>
	__host__ __device__
        const T *Matrix <T> ::operator[](size_t i) const
        {
                return (this->__array + i * __cols);
        }
        
	template <class T>
	__host__ __device__
	void Matrix <T> ::set_all(T x)
	{
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = x;
	}

        template <class T>
	__host__ __device__
        size_t Matrix <T> ::get_rows() const
        {
                return __rows;
        }

        template <class T>
	__host__ __device__
        size_t Matrix <T> ::get_cols() const
        {
                return __cols;
        }

        template <class T>
	__host__ __device__
        Matrix <T> Matrix <T> ::transpose() const
        {
                return Matrix <T> (__cols, __rows,
			[this] __host__ __device__ (size_t i, size_t j) {
				return this->__array[j * __cols + i];
			}
		);
        }

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
        
	template <class T>
	__host__ __device__
        void Matrix <T> ::operator*=(const T &x)
        {
                for (size_t i = 0; i < this->__size; i++)
			this->__array[i] *= x;
        }
       
	template <class T>
	__host__ __device__
        void Matrix <T> ::operator/=(const T &x)
        {
                for (size_t i = 0; i < this->__size; i++)
			this->__array[i] /= x;
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

        template <class T>
	__host__ __device__
        Matrix <T> operator*(const Matrix <T> &a, const Matrix <T> &b)
        {

#ifndef __CUDA_ARCH__

		if (a.__cols != b.__rows)
			throw typename Matrix <T> ::dimension_mismatch();

#endif

                return Matrix <T> (a.__rows, b.__cols,
			[a, b] __host__ __device__ (size_t i, size_t j) {
				T acc = 0;

				for (size_t k = 0; k < a.__cols; k++)
					acc += a[i][k] * b[k][j];

				return acc;
			}
		);
        }

        template <class T>
	__host__ __device__
        Matrix <T> operator*(const Matrix <T> &a, const T &scalar)
        {
                return Matrix <T> (a.__rows, a.__cols,
			[&](size_t i, size_t j) {
				return a[i][j] * scalar;
			}
		);
        }

        template <class T>
	__host__ __device__
        Matrix <T> operator*(const T &scalar, const Matrix <T> &a)
        {
                return a * scalar;
        }

        template <class T>
	__host__ __device__
        Matrix <T> operator/(const Matrix <T> &a, const T &scalar)
        {
                return Matrix <T> (a.__rows, a.__cols,
			[&](size_t i, size_t j) {
                        	return a[i][j] / scalar;
                	}
		);
        }

        template <class T>
	__host__ __device__
        Matrix <T> operator/(const T &scalar, const Matrix <T> &a)
        {
                return a / scalar;
        }

        template <class T>
	__host__ __device__
        bool operator==(const Matrix <T> &a, const Matrix <T> &b)
        {
                if ((a.__rows != b.__rows) || (a.__cols != b.__cols))
                        return false;

                for (size_t i = 0; i  < a.__rows; i++) {
                        for (size_t j = 0; j < a.__cols; j++) {
                                if (a[i][j] != b[i][j])
                                        return false;
                        }
                }

                return true;
        }

        template <class T>
	__host__ __device__
        Matrix <T> shur(const Matrix <T> &a, const Matrix <T> &b)
        {

#ifndef __CUDA_ARCH__
                
		// TODO: Replace with better exception handling for the device
		if (!((a.get_rows() == b.get_rows())
			&& (a.get_cols() == b.get_cols())))
			throw typename Matrix <T> ::dimension_mismatch();

                return Matrix <T> (a.get_rows(), b.get_cols(),
			[&](size_t i, size_t j) {
                        	return a[i][j] * b[i][j];
			}
		);

#else

                return Matrix <T> (a.__rows, a.__cols,
			[a, b] __device__ (size_t i, size_t j) {
                        	return a[i][j] * b[i][j];
			}
		);

#endif

        }

}

#endif
