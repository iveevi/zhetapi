#ifndef VECTOR_CUH_
#define VECTOR_CUH_

#define ZHP_CUDA

#include <vector.hpp>

// CUDA headers
#include <cuda/matrix.cuh>

namespace zhetapi {
	
	size_t cpu_vector_copies = 0;
	__device__ size_t gpu_vector_copies = 0;

	template <class T>
	__host__ __device__
	Vector <T> ::Vector() : Matrix <T> () {}

	template <class T>
	__host__ __device__
	Vector <T> ::Vector(const Vector &other) : Matrix <T> (other.size(), 1, T())
	{

#ifdef __CUDA_ARCH__

		gpu_vector_copies++;

#else

		cpu_vector_copies++;

#endif
		
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = other.__array[i];
	}

	template <class T>
	__host__ __device__
	Vector <T> ::Vector(const Matrix <T> &other) : Matrix <T> (other.get_rows(), 1, T())
	{
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = other[0][i];
	}

	template <class T>
	__host__ __device__
	Vector <T> ::Vector(size_t rs, T def) : Matrix <T> (rs, 1, def) {}

	// FIXME: Delegate Matrix constructor
	template <class T>
	__host__ __device__
	Vector <T> ::Vector(size_t rs, T *arr) : Matrix <T> (rs, 1, arr) {}

	template <class T>
	template <class F>
	__host__ __device__
	Vector <T> ::Vector(size_t rs, F gen)
		: Matrix <T> (rs, 1, T())
	{
		for (int i = 0; i < this->__size; i++)
			this->__array[i] = gen(i);
	}

	template <class T>
	__host__ __device__
	Vector <T> &Vector <T> ::operator=(const Vector <T> &other)
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
			
			this->__dims = 1;
			this->__dim = new size_t[1];

			this->__dim[0] = this->__size;
		}

		return *this;
	}

	template <class T>
	__host__ __device__
	Vector <T> &Vector <T> ::operator=(const Matrix <T> &other)
	{
		if (this != &other) {
			*this = Vector(other.get_rows(), T());

			for (size_t i = 0; i < this->__size; i++)
				this->__array[i] = other[i][0];
		}

		return *this;
	}
	
	// TODO: Turn this method into a virtual method with the base as Tensor
	template <class T>
	void Vector <T> ::copy_to_device(const Vector <T> &other)
	{
		/* printf("Copying vector to device...\n");
		printf("\tOther.size = %d ({%f, %f})\n", other.__size, other[0],
				other[1]); */

		this->__on_device = true;

		if (this->__array)
			delete[] this->__array;

		if (this->__dim)
			delete[] this->__dim;

		cudaMalloc(&(this->__array), sizeof(T) * other.__size);
		cudaMemcpy(this->__array, other.__array, sizeof(T) *
				other.__size, cudaMemcpyHostToDevice);

		this->__rows = other.__rows;
		this->__cols = other.__cols;

		this->__size = other.__size;

		this->__dims = 2;

		cudaMalloc(&this->__dim, 2 * sizeof(size_t));
		cudaMemcpy(this->__dim, other.__dim, 2 * sizeof(size_t),
				cudaMemcpyHostToDevice);
	}

	// TODO: Turn this method into a virtual method with the base as Tensor
	template <class T>
	void Vector <T> ::transfer_from_device(Vector <T> &other)
	{
		if (other.__array)
			delete[] other.__array;

		if (other.__dim)
			delete[] other.__dim;

		other.__rows = this->__rows;
		other.__cols = this->__cols;

		other.__size = this->__rows * this->__cols;
		
		other.__array = new T[this->__size];

		cudaMemcpy(other.__array, this->__array, sizeof(T) *
				this->__size, cudaMemcpyDeviceToHost);
		cudaCheckError(this->__array);

		other.__dims = 2;

		other.__dim = new size_t[2];
		other.__dim[0] = this->__rows;
		other.__dim[1] = this->__cols;
	}
	
	// Indexing operators
	template <class T>
	__host__ __device__
	T &Vector <T> ::operator[](size_t i)
	{
		return this->__array[i];
	}

	template <class T>
	__host__ __device__
	const T &Vector <T> ::operator[](size_t i) const
	{
		return this->__array[i];
	}

	template <class T>
	__host__ __device__
	size_t Vector <T> ::size() const
	{
		return this->__size;
	}

	template <class T>
	__host__ __device__
	void Vector <T> ::operator+=(const Vector <T> &a)
	{
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] += a.__array[i];
	}

	template <class T>
	__host__ __device__
	void Vector <T> ::operator-=(const Vector <T> &a)
	{
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] -= a.__array[i];
	}

	template <class T>
	__host__ __device__
	Vector <T> Vector <T> ::append_above(const T &x)
	{
		T *apped = new T[this->__size + 1];

		apped[0] = x;
		for (size_t i = 1; i <= this->__size; i++)
			apped[i] = this->__array[i - 1];

		return Vector(this->__size + 1, apped);
	}

	template <class T>
	__host__ __device__
	Vector <T> Vector <T> ::append_below(const T &x)
	{
		size_t t_sz = size();

		::std::vector <T> total;

		for (size_t i = 0; i < t_sz; i++)
			total.push_back((*this)[i]);

		total.push_back(x);

		return Vector(total);
	}

	template <class T>
	__host__ __device__
	Vector <T> Vector <T> ::remove_top()
	{
		T *apped = new T[this->__size - 1];

		for (size_t i = 0; i < this->__size - 1; i++)
			apped[i] = this->__array[i + 1];

		return Vector(this->__size - 1, apped);
	}

	template <class T>
	__host__ __device__
	Vector <T> Vector <T> ::remove_bottom()
	{
		size_t t_sz = size();

		::std::vector <T> total;
		for (size_t i = 0; i < t_sz - 1; i++)
			total.push_back((*this)[i]);

		return Vector(total);
	}

	// Miscellaneous functions
	template <class T>
	T Vector <T> ::norm() const
	{
		return sqrt(inner(*this, *this));
	}
	
	/* template <class T>
	__host__ __device__
        Vector <T> operator-(const Vector <T> &a, const Vector <T> &b)
        {
                assert(a.__rows == b.__rows && a.__cols == b.__cols);
		return Matrix <T> (a.__rows, a.__cols,
			[a, b] __device__ (size_t i, size_t j) {
                        	return a[i][j] - b[i][j];
			}
		);
        } */
	
	template <class T>
	__host__ __device__
	T inner(const Vector <T> &a, const Vector <T> &b)
	{
		T acc = 0;

		assert(a.size() == b.size());
		for (size_t i = 0; i < a.__size; i++)
			acc += a[i] * b[i];

		return acc;
	}

}

#endif
