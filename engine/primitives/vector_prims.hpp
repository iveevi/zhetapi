// Constructors
template <class T>
Vector <T> ::Vector() : Matrix <T> () {}

template <class T>
Vector <T> ::Vector(size_t len)
		: Matrix <T> (len, 1) {}

template <class T>
Vector <T> ::Vector(size_t rs, T def)
		: Matrix <T> (rs, 1, def) {}

#ifdef __AVR

template <class T>
Vector <T> ::Vector(size_t rs, T (*gen)(size_t))
		: Matrix <T> (rs, 1, gen) {}

template <class T>
Vector <T> ::Vector(size_t rs, T *(*gen)(size_t))
		: Matrix <T> (rs, 1, gen) {}

#endif

template <class T>
Vector <T> ::Vector(size_t rs, T *ref, bool slice)
		: Matrix <T> (rs, 1, ref, slice) {}

// Copy constructors
template <class T>
Vector <T> ::Vector(const Vector &other)
		: Matrix <T> (other.size(), 1, T())
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] = other._array[i];
}

template <class T>
Vector <T> ::Vector(const Matrix <T> &other)
		: Matrix <T> (other.get_rows(), 1, T())
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] = other[0][i];
}

template <class T>
T &Vector <T> ::x()
{
	if (this->_size < 1)
		throw index_out_of_bounds();

	return this->_array[0];
}

template <class T>
T &Vector <T> ::y()
{
	if (this->_size < 2)
		throw index_out_of_bounds();

	return this->_array[1];
}

template <class T>
T &Vector <T> ::z()
{
	if (this->_size < 3)
		throw index_out_of_bounds();

	return this->_array[2];
}

template <class T>
const T &Vector <T> ::x() const
{
	if (this->_size < 1)
		throw index_out_of_bounds();

	return this->_array[0];
}

template <class T>
const T &Vector <T> ::y() const
{
	if (this->_size < 2)
		throw index_out_of_bounds();

	return this->_array[1];
}

template <class T>
const T &Vector <T> ::z() const
{
	if (this->_size < 3)
		throw index_out_of_bounds();

	return this->_array[2];
}

template <class T>
T &Vector <T> ::operator[](size_t i)
{
	return this->_array[i];
}

template <class T>
const T &Vector <T> ::operator[](size_t i) const
{
	return this->_array[i];
}

template <class T>
size_t Vector <T> ::size() const
{
	return this->_size;
}

template <class T>
T Vector <T> ::arg() const
{
	return atan2((*this)[1], (*this)[0]);
}

template <class T>
T Vector <T> ::min() const
{
	T mn = this->_array[0];

	for (size_t j = 1; j < this->_size; j++) {
		if (mn > this->_array[j])
			mn = this->_array[j];
	}

	return mn;
}

template <class T>
T Vector <T> ::max() const
{
	T mx = this->_array[0];

	for (size_t j = 1; j < this->_size; j++) {
		if (mx < this->_array[j])
			mx = this->_array[j];
	}

	return mx;
}

template <class T>
size_t Vector <T> ::imin() const
{
	size_t i = 0;

	for (size_t j = 1; j < this->_size; j++) {
		if (this->_array[i] > this->_array[j])
			i = j;
	}

	return i;
}

template <class T>
size_t Vector <T> ::imax() const
{
	size_t i = 0;

	for (size_t j = 1; j < this->_size; j++) {
		if (this->_array[i] < this->_array[j])
			i = j;
	}

	return i;
}

template <class T>
void Vector <T> ::normalize()
{
	T dt = this->norm();

	for (size_t i = 0; i < size(); i++)
		(*this)[i] /= dt;
}

// Non-member operators
template <class T>
Vector <T> operator+(const Vector <T> &a, const Vector <T> &b)
{
	Vector <T> out = a;

	out += b;

	return out;
}

template <class T>
Vector <T> operator-(const Vector <T> &a, const Vector <T> &b)
{
	Vector <T> out = a;

	out -= b;

	return out;
}

template <class T>
Vector <T> operator*(const Vector <T> &a, const T &b)
{
	Vector <T> out = a;

	for (size_t i = 0; i < a.size(); i++)
		out[i] *= b;

	return out;
}

template <class T>
Vector <T> operator*(const T &b, const Vector <T> &a)
{
	Vector <T> out = a;

	for (size_t i = 0; i < a.size(); i++)
		out[i] *= b;

	return out;
}

template <class T>
Vector <T> operator/(const Vector <T> &a, const T &b)
{
	Vector <T> out = a;

	for (size_t i = 0; i < a.size(); i++)
		out[i] /= b;

	return out;
}

template <class T>
Vector <T> operator/(const T &b, const Vector <T> &a)
{
	Vector <T> out = a;

	for (size_t i = 0; i < a.size(); i++)
		out[i] /= b;

	return out;
}

// Static methods
template <class T>
Vector <T> Vector <T> ::one(size_t size)
{
	return Vector <T> (size, T(1));
}

template <class T>
Vector <T> Vector <T> ::rarg(double r, double theta)
{
	return Vector <T> {r * cos(theta), r * sin(theta)};
}

// Non-member functions
template <class F, class T>
T max(F ftn, const Vector <T> &values)
{
	T max = ftn(values[0]);

	size_t n = values.size();
	for (size_t i = 1; i < n; i++) {
		T k = ftn(values[i]);
		if (k > max)
			max = k;
	}

	return max;
}

template <class F, class T>
T min(F ftn, const Vector <T> &values)
{
	T min = ftn(values[0]);

	size_t n = values.size();
	for (size_t i = 1; i < n; i++) {
		T k = ftn(values[i]);
		if (k < min)
			min = k;
	}

	return min;
}

template <class F, class T>
T argmax(F ftn, const Vector <T> &values)
{
	T max = values[0];

	size_t n = values.size();
	for (size_t i = 1; i < n; i++) {
		if (ftn(values[i]) > ftn(max))
			max = values[i];
	}

	return max;
}

template <class F, class T>
T argmin(F ftn, const Vector <T> &values)
{
	T min = values[0];

	size_t n = values.size();
	for (size_t i = 1; i < n; i++) {
		if (ftn(values[i]) < ftn(min))
			min = values[i];
	}

	return min;
}

template <class T>
Vector <T> cross(const Vector <T> &a, const Vector <T> &b)
{
	// Switch between 2 and 3
	assert((a._size == 3) && (a._size == 3));

	return Vector <T> {
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]
	};
}

template <class T>
Vector <T> concat(const Vector <T> &a, const Vector <T> &b)
{
	T *arr = new T[a._dim[0] + b._dim[0]];

	for (size_t i = 0; i < a.size(); i++)
		arr[i] = a[i];
	
	for (size_t i = 0; i < b.size(); i++)
		arr[a.size() + i] = b[i];
	
	return Vector <T> (a.size() + b.size(), arr);
}

template <class T, class ... U>
Vector <T> concat(const Vector <T> &a, const Vector <T> &b, U ... args)
{
	return concat(concat(a, b), args...);
}

template <class T>
T inner(const Vector <T> &a, const Vector <T> &b)
{
	T acc = 0;

	assert(a.size() == b.size());
	for (size_t i = 0; i < a._size; i++)
		acc += a[i] * b[i];

	return acc;
}

template <class T, class U>
T inner(const Vector <T> &a, const Vector <U> &b)
{
	T acc = 0;

	assert(a.size() == b.size());
	for (size_t i = 0; i < a._size; i++)
		acc += (T) (a[i] * b[i]);	// Cast the result

	return acc;
}

// Externally defined methods
template <class T>
Vector <T> Tensor <T> ::cast_to_vector() const
{
	// Return a slice-vector
	return Vector <T> (_size, _array);
}
