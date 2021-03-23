template <class T>
Tensor <T> ::Tensor() {}

template <class T>
Tensor <T> ::Tensor(const Tensor <T> &other)
		: __size(other.__size),
		__dims(other.__dims)
{
	__dim = new size_t[__dims];
	memcpy(__dim, other.__dim, sizeof(size_t) * __dims);

	__array = new T[__size];
	memcpy(__array, other.__array, sizeof(T) * __size);
}

template <class T>
Tensor <T> ::Tensor(size_t rows, size_t cols)
		: __size(rows * cols),
		__dims(2)
{
	__dim = new size_t[2];
	__dim[0] = rows;
	__dim[1] = cols;

	__array = new T[__size];
}

template <class T>
Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
{
	if (this != &other) {
		__dims = other.__dims;
		__size = other.__size;
		
		__dim = new size_t[__dims];
		memcpy(__dim, other.__dim, sizeof(size_t) * __dims);

		__array = new T[__size];
		memcpy(__array, other.__array, sizeof(T) * __size);
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
	if (!__array && !__dim)
		return;

	if (__dim)
		delete[] __dim;

	if (__array && !__sliced)
		delete[] __array;

	__array = nullptr;
	__dim = nullptr;
}

template <class T>
size_t Tensor <T> ::size() const
{
	return __size;
}


// Comparison
template <class T>
bool operator==(const Tensor <T> &a, const Tensor <T> &b)
{
	if (a.__size != b.__size)
		return false;

	for (size_t i = 0; i < a.__size; i++) {
		if (a.__array[i] != b.__array[i])
			return false;
	}

	return true;
}

template <class T>
bool operator!=(const Tensor <T> &a, const Tensor <T> &b)
{
	return !(a == b);
}
