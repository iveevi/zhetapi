#ifndef VECTOR_PRIMITIVES_H_
#define VECTOR_PRIMITIVES_H_

namespace zhetapi {

/**
 * @brief Default vector constructor.
 */
template <class T>
Vector <T> ::Vector() : Matrix <T> () {}

/**
 * @brief Size constructor. Components are initialized to 0 or the default value
 * of T.
 *
 * @param len size of the vector.
 */
template <class T>
Vector <T> ::Vector(size_t len)
		: Matrix <T> (len, 1) {}

/**
 * @brief Size constructor. Each component is initialized to def.
 *
 * @param rs the number of rows (size) of the vector.
 * @param def the value each component is initialized to.
 */
template <class T>
Vector <T> ::Vector(size_t rs, T def)
		: Matrix <T> (rs, 1, def) {}

#ifdef __AVR

/**
 * @brief Size constructor. Each component is evaluated from a function which
 * depends on the index.
 *
 * @param rs the number of rows (size) of the vector.
 * @param gen a pointer to the function that generates the coefficients.
 */
template <class T>
Vector <T> ::Vector(size_t rs, T (*gen)(size_t))
		: Matrix <T> (rs, 1, gen) {}

/**
 * @brief Size constructor. Each component is evaluated from a function which
 * depends on the index.
 *
 * @param rs the number of rows (size) of the vector.
 * @param gen a pointer to the function that generates pointers to the
 * coefficients.
 */
template <class T>
Vector <T> ::Vector(size_t rs, T *(*gen)(size_t))
		: Matrix <T> (rs, 1, gen) {}

#endif

template <class T>
Vector <T> ::Vector(size_t rs, T *ref, bool slice)
		: Matrix <T> (rs, 1, ref, slice) {}

/**
 * @brief Copy constructor.
 * 
 * @param other the reference vector (to be copied from).
 */
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

// Assignment operators
template <class T>
Vector <T> &Vector <T> ::operator=(const Vector <T> &other)
{
	if (this != &other) {
		this->clear();

		this->_array = new T[other._size];
		this->_rows = other._rows;
		this->_cols = other._cols;

		this->_size = other._size;
		for (size_t i = 0; i < this->_size; i++)
			this->_array[i] = other._array[i];
		
		this->_dims = 1;
		this->_dim = new size_t[1];

		this->_dim[0] = this->_size;
	}

	return *this;
}

template <class T>
Vector <T> &Vector <T> ::operator=(const Matrix <T> &other)
{
	if (this != &other) {
		*this = Vector(other.get_rows(), T());

		for (size_t i = 0; i < this->_size; i++)
			this->_array[i] = other[0][i];
	}

	return *this;
}

/**
 * @return the first component of the vector (index 0).
 */
template <class T>
T &Vector <T> ::x()
{
	if (this->_size < 1)
		throw index_out_of_bounds();

	return this->_array[0];
}

/**
 * @return the second component of the vector (index 1).
 */
template <class T>
T &Vector <T> ::y()
{
	if (this->_size < 2)
		throw index_out_of_bounds();

	return this->_array[1];
}

/**
 * @return the third component of the vector (index 2).
 */
template <class T>
T &Vector <T> ::z()
{
	if (this->_size < 3)
		throw index_out_of_bounds();

	return this->_array[2];
}

/**
 * @return the first component of the vector (index 0).
 */
template <class T>
const T &Vector <T> ::x() const
{
	if (this->_size < 1)
		throw index_out_of_bounds();

	return this->_array[0];
}

/**
 * @return the second component of the vector (index 1).
 */
template <class T>
const T &Vector <T> ::y() const
{
	if (this->_size < 2)
		throw index_out_of_bounds();

	return this->_array[1];
}

/**
 * @return the third component of the vector (index 2).
 */
template <class T>
const T &Vector <T> ::z() const
{
	if (this->_size < 3)
		throw index_out_of_bounds();

	return this->_array[2];
}

/**
 * @brief Indexing operator
 *
 * @param i the specified index
 *
 * @return the \f$i\f$th component of the vector.
 */
template <class T>
T &Vector <T> ::operator[](size_t i)
{
	return this->_array[i];
}

/**
 * @brief Indexing operator
 *
 * @param i the specified index
 *
 * @return the \f$i\f$th component of the vector.
 */
template <class T>
const T &Vector <T> ::operator[](size_t i) const
{
	return this->_array[i];
}

/**
 * @return the size of the vector.
 */
template <class T>
size_t Vector <T> ::size() const
{
	return this->_size;
}

/**
 * @brief Returns the argument of the vector. Assumes that the vector has at
 * least two components.
 *
 * @return the argument of the vector in radians (the angle at which the vector
 * is pointing to).
 */
template <class T>
T Vector <T> ::arg() const
{
	return atan2(y(), x());
}

/**
 * @brief The minimum component of the vector.
 * 
 * @return the smallest component, \f$\min v_i.\f$
 */
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

/**
 * @brief The maximum component of the vector.
 * 
 * @return the largest component, \f$\max v_i.\f$
 */
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

/**
 * @brief The index of the smallest component: essentially argmin.
 * 
 * @return the index of the smallest component.
 */
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

/**
 * @brief The index of the largest component: essentially argmax.
 * 
 * @return the index of the largest component.
 */
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

/**
 * @brief Normalizes the components of the vector (the modified vector will have
 * unit length).
 */
template <class T>
void Vector <T> ::normalize()
{
	T dt = this->norm();

	for (size_t i = 0; i < size(); i++)
		(*this)[i] /= dt;
}

// TODO: rename these functions (or add): they imply modification (also add const)
// TODO: use memcpy later
template <class T>
Vector <T> Vector <T> ::append_above(const T &x) const
{
	T *arr = new T[size() + 1];
	arr[0] = x;
	for (size_t i = 0; i < size(); i++)
		arr[i + 1] = this->_array[i];
	return Vector(size() + 1, arr, false);
}

template <class T>
Vector <T> Vector <T> ::append_below(const T &x)
{
	T *arr = new T[size() + 1];
	for (size_t i = 0; i < size(); i++)
		arr[i] = this->_array[i];
	arr[size()] = x;
	return Vector(size() + 1, arr, false);
}

template <class T>
Vector <T> Vector <T> ::remove_top()
{
	T *arr = new T[size() - 1];
	for (size_t i = 1; i < size(); i++)
		arr[i - 1] = this->_array[i];
	return Vector(size() - 1, arr, false);
}

template <class T>
Vector <T> Vector <T> ::remove_bottom()
{
	T *arr = new T[size() - 1];
	for (size_t i = 0; i < size() - 1; i++)
		arr[i] = this->_array[i];
	return Vector(size() - 1, arr, false);
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

}

#endif
