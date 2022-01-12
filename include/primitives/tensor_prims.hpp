#ifndef TENSOR_PRIMITIVES_H_
#define TENSOR_PRIMITIVES_H_

#ifdef __CUDACC__

#include "cuda/nvarena.cuh"

#endif

namespace zhetapi {

/**
 * @brief Default constructor.
 */
template <class T>
Tensor <T> ::Tensor() {}

/**
 * @brief Homogenous (with respect to the component type) copy constructor.
 *
 * @param other the reference vector (to be copied from).
 */
template <class T>
Tensor <T> ::Tensor(const Tensor <T> &other)
		: _size(other._size), _dims(other._dims)
{
	// Faster for homogenous types
	_dim = new size_t[_dims];
	memcpy(_dim, other._dim, sizeof(size_t) * _dims);

	_array = new T[_size];
	memcpy(_array, other._array, sizeof(T) * _size);
}

/**
 * @brief Heterogenous (with respect to the component type) copy constructor.
 *
 * @param other the reference vector (to be copied from).
 */
template <class T>
template <class A>
Tensor <T> ::Tensor(const Tensor <A> &other)
		: _size(other._size), _dims(other._dims)
{
	_dim = new size_t[_dims];
	memcpy(_dim, other._dim, sizeof(size_t) * _dims);

	_array = new T[_size];
	for (size_t i = 0; i < _size; i++)
		_array[i] = static_cast <T> (other._array[i]);
}

template <class T>
Tensor <T> ::Tensor(size_t rows, size_t cols)
		: _dims(2), _size(rows * cols)
{
	_dim = new size_t[2];
	_dim[0] = rows;
	_dim[1] = cols;

	_array = new T[_size];
}

/**
 * @brief Full slice constructor. Makes a Tensor out of existing (previously
 * allocated memory), and the ownership can be decided. By default, the Tensor
 * does not gain ownership over the memory. Note that the sizes given are not
 * checked for validity.
 *
 * @param dims the number of dimensions to slice.
 * @param dim the dimension size array.
 * @param size the size of the Tensor.
 * @param array the components of the Tensor.
 * @param slice the slice flag. Set to \c true to make sure the memory is not
 * deallocated by the resulting Tensor, and \c false otherwise.
 */
template <class T>
Tensor <T> ::Tensor(size_t dims, size_t *dim, size_t size, T *array, bool slice)
		: _dims(dims), _dim(dim), _size(size), _array(array),
		_dim_sliced(slice), _arr_sliced(slice) {}

template <class T>
template <class A>
Tensor <T> &Tensor <T> ::operator=(const Tensor <A> &other)
{
	if (this != &other) {
		_dims = other._dims;
		_size = other._size;

		_dim = new size_t[_dims];
		memcpy(_dim, other._dim, sizeof(size_t) * _dims);

		_array = new T[_size];
		for (size_t i = 0; i < _size; i++)
			_array[i] = static_cast <T> (other._array[i]);
	}

	return *this;
}

template <class T>
Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
{
	// Faster version for homogenous types (memcpy is faster)
	if (this != &other) {
		_dims = other._dims;
		_size = other._size;

		_dim = new size_t[_dims];
		memcpy(_dim, other._dim, sizeof(size_t) * _dims);

		_array = new T[_size];
		memcpy(_array, other._array, sizeof(T) * _size);
	}

	return *this;
}

/**
 * @brief Deconstructor.
 */
template <class T>
Tensor <T> ::~Tensor()
{
	clear();
}

template <class T>
void Tensor <T> ::clear()
{
	if (!_array && !_dim)
		return;

	if (!_dim_sliced) {

#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
		if (_on_device) {
			if (!_arena)
				throw null_nvarena();

			_arena->free(_dim);
		} else {
			delete[] _dim;
		}
#else
		delete[] _dim;
#endif

	}

	if (!_arr_sliced) {

#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
		if (_on_device) {
			if (!_arena)
				throw null_nvarena();

			_arena->free(_array);
		} else {
			delete[] _array;
		}
#else
		delete[] _array;
#endif

	}

	_array = nullptr;
	_dim = nullptr;
}

/**
 * @brief Returns the size of the tensor.
 *
 * @return the size of the tensor (number of components in the tensor).
 */
template <class T>
__cuda_dual__
size_t Tensor <T> ::size() const
{
	return _size;
}

/**
 * @brief Returns the number of dimensions in the tensor.
 *
 * @return the number of dimensions in the tensor.
 */
template <class T>
__cuda_dual__
size_t Tensor <T> ::dimensions() const
{
	return _dims;
}

/**
 * @brief Returns the size of a specific dimension. Does not check bounds.
 *
 * @param i the desired index.
 *
 * @return the size of dimension \p i.
 */
template <class T>
__cuda_dual__
size_t Tensor <T> ::dim_size(size_t i) const
{
	return _dim[i];
}

/**
 * @brief Returns the size of a specific dimension. If the index is out of
 * bounds of the number of dimensions, then 1 is returned.
 *
 * @param i the desired index.
 *
 * @return the size of dimension \p i.
 */
template <class T>
__cuda_dual__
size_t Tensor <T> ::safe_dim_size(size_t i) const
{
	return (_dims > i) ? _dim[i] : 1;
}

// Comparison
template <class T>
bool operator==(const Tensor <T> &a, const Tensor <T> &b)
{
	if (a._size != b._size)
		return false;

	for (size_t i = 0; i < a._size; i++) {
		if (a._array[i] != b._array[i])
			return false;
	}

	return true;
}

template <class T>
bool operator!=(const Tensor <T> &a, const Tensor <T> &b)
{
	return !(a == b);
}

}

#endif
