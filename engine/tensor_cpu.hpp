template <class T>
Tensor <T> ::Tensor(const std::vector <size_t> &dim, const std::vector <T> &arr)
		: __dims(dim.size())
{

#ifdef __CUDA_ARCH__

	__on_device = false;

#endif

	__dim = new size_t[__dims];

	size_t prod = 1;
	for (size_t i = 0; i < __dims; i++) {
		prod *= dim[i];

		__dim[i] = dim[i];
	}

	__size = prod;

	if (__size <= 0)
		throw bad_dimensions();

	if (arr.size() != __size)
		throw dimension_mismatch();

	__array = new T[prod];

	for (size_t i = 0; i < prod; i++)
		__array[i] = arr[i];
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
bool Tensor <T> ::good() const
{
	return __array != nullptr;
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

// Arithmetic
template <class T>
void Tensor <T> ::operator*=(const T &x)
{
	for (size_t i = 0; i < __size; i++)
		__array[i] *= x;
}

template <class T>
void Tensor <T> ::operator/=(const T &x)
{
	for (size_t i = 0; i < __size; i++)
		__array[i] /= x;
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
	os << print(ts.__array, ts.__size, ts.__dim, 0, ts.__dims - 1);

	return os;
}
