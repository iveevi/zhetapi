#ifndef TENSOR_CPU_H_
#define TENSOR_CPU_H_

// Constructors and memory relevant functions
template <class T>
Tensor <T> ::Tensor() : __dim(nullptr), __dims(0), __array(nullptr),
		__size(0) {}

template <class T>
Tensor <T> ::Tensor(const Tensor <T> &other) : __dims(other.__dims), __size(other.__size)
{
	__dim = new size_t[__dims];
	for (size_t i = 0; i < __dims; i++)
		__dim[i] = other.__dim[i];

	__array = new T[__size];
	for (size_t i = 0; i < __size; i++)
		__array[i] = other.__array[i];
}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim)
		: __dims(dim.size())
{
	__dim = new size_t[__dims];

	size_t prod = 1;
	for (size_t i = 0; i < __dims; i++) {
		prod *= dim[i];

		__dim[i] = dim[i];
	}

	__size = prod;

	if (!__size)
		return;

	__array = new T[prod];
}

template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim, const T &def)
		: __dims(dim.size())
{
	__dim = new size_t[__dims];

	size_t prod = 1;
	for (size_t i = 0; i < __dims; i++) {
		prod *= dim[i];

		__dim[i] = dim[i];
	}

	__size = prod;

	if (!__size)
		return;

	__array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		__array[i] = def;
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
bool Tensor <T> ::good() const
{
	return __array != nullptr;
}

template <class T>
Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
{
	if (this != &other) {
		__dims = other.__dims;
		__size = other.__size;
	
		__dim = new size_t[__dims];
		for (size_t i = 0; i < __dims; i++)
			__dim[i] = other.__dim[i];

		__array = new T[__size];
		for (size_t i = 0; i < __size; i++)
			__array[i] = other.__array[i];
	}

	return *this;
}

// Actions
template <class T>
void Tensor <T> ::nullify(long double p, const Interval <1> &i)
{
	for (size_t k = 0; k < __size; k++) {
		if (p > i.uniform())
			__array[k] = T(0);
	}
}

// Index
template <class T>
T &Tensor <T> ::operator[](const std::vector <size_t> &indices)
{
	size_t full = 0;

	assert(indices.size() == __dims);
	for (size_t i = 0; i < __dims; i++)
		full += indices[i] * __dim[__dims - (i + 1)];
	
	return __array[full];
}

template <class T>
const T &Tensor <T> ::operator[](const ::std::vector <size_t> &indices) const
{
	size_t full = 0;

	assert(indices.size() == __dims);
	for (size_t i = 0; i < __dims; i++)
		full += indices[i] * __dim[__dims - (i + 1)];
	
	return __array[full];
}

#endif