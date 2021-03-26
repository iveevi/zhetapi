template <class T>
Tensor <T> ::Tensor() {}

template <class T>
Tensor <T> ::Tensor(const Tensor <T> &other)
		: _size(other._size),
		_dims(other._dims)
{
	_dim = new size_t[_dims];
	memcpy(_dim, other._dim, sizeof(size_t) * _dims);

	_array = new T[_size];
	memcpy(_array, other._array, sizeof(T) * _size);
}

template <class T>
Tensor <T> ::Tensor(size_t rows, size_t cols)
		: _size(rows * cols),
		_dims(2)
{
	_dim = new size_t[2];
	_dim[0] = rows;
	_dim[1] = cols;

	_array = new T[_size];
}

template <class T>
Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
{
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

	if (_dim)
		delete[] _dim;

	if (_array && !_sliced)
		delete[] _array;

	_array = nullptr;
	_dim = nullptr;
}

template <class T>
size_t Tensor <T> ::size() const
{
	return _size;
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
