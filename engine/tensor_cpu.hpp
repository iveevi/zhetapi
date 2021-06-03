#ifndef TENSOR_CPU_H_
#define TENSOR_CPU_H_

namespace zhetapi {

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim, const std::vector <T> &arr)
		: _dims(dim.size())
{

#ifdef _CUDA_ARCH_

	_on_device = false;

#endif

	_dim = new size_t[_dims];

	size_t prod = 1;
	for (size_t i = 0; i < _dims; i++) {
		prod *= dim[i];

		_dim[i] = dim[i];
	}

	_size = prod;

	if (_size <= 0)
		throw bad_dimensions();

	if (arr.size() != _size)
		throw dimension_mismatch();

	_array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		_array[i] = arr[i];
}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim)
		: _dims(dim.size())
{
	_dim = new size_t[_dims];

	size_t prod = 1;
	for (size_t i = 0; i < _dims; i++) {
		prod *= dim[i];

		_dim[i] = dim[i];
	}

	_size = prod;

	if (!_size)
		return;

	_array = new T[prod];
}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim, const T &def)
		: _dims(dim.size())
{
	_dim = new size_t[_dims];

	size_t prod = 1;
	for (size_t i = 0; i < _dims; i++) {
		prod *= dim[i];

		_dim[i] = dim[i];
	}

	_size = prod;

	if (!_size)
		return;

	_array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		_array[i] = def;
}

template <class T>
bool Tensor <T> ::good() const
{
	return _array != nullptr;
}

// Actions
template <class T>
void Tensor <T> ::nullify(long double p, const Interval <1> &i)
{
	for (size_t k = 0; k < _size; k++) {
		if (p > i.uniform())
			_array[k] = T(0);
	}
}

// Index
template <class T>
T &Tensor <T> ::operator[](const std::vector <size_t> &indices)
{
	size_t full = 0;

	assert(indices.size() == _dims);
	for (size_t i = 0; i < _dims; i++)
		full += indices[i] * _dim[_dims - (i + 1)];
	
	return _array[full];
}

template <class T>
const T &Tensor <T> ::operator[](const std::vector <size_t> &indices) const
{
	size_t full = 0;

	assert(indices.size() == _dims);
	for (size_t i = 0; i < _dims; i++)
		full += indices[i] * _dim[_dims - (i + 1)];
	
	return _array[full];
}

// Arithmetic
template <class T>
void Tensor <T> ::operator*=(const T &x)
{
	for (size_t i = 0; i < _size; i++)
		_array[i] *= x;
}

template <class T>
void Tensor <T> ::operator/=(const T &x)
{
	for (size_t i = 0; i < _size; i++)
		_array[i] /= x;
}

// Printing functions
template <class T>
std::string print(T *arr, size_t size, size_t *ds, size_t dn, size_t dmax)
{
	if (size == 0)
		return "[]";
	
	std::string out = "[";

	// Size of each dimension
	size_t dsize = size / ds[dn];

	T *current = arr;
	for (size_t i = 0; i < ds[dn]; i++) {
		if (dn == dmax)
			out += std::to_string(*current);
		else
			out += print(current, dsize, ds, dn + 1, dmax);

		if (i < ds[dn] - 1)
			out += ", ";

		current += dsize;
	}

	return out + "]";
}

template <class T>
std::ostream &operator<<(std::ostream &os, const Tensor <T> &ts)
{
	os << print(ts._array, ts._size, ts._dim, 0, ts._dims - 1);

	return os;
}

}

#endif