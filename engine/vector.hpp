#ifndef ELEMENT_H_
#define ELEMENT_H_

// Engine headers
#include <complex.hpp>
#include <matrix.hpp>
#include <rational.hpp>

/**
 * @brief Representative
 * of a general Vector,
 * with components of type
 * T. In relation to a Matrix,
 * these are only column vectors:
 * transpose them to get the
 * corresponding row vectors.
 */
template <class T>
class Vector : public Matrix <T> {
public:
	Vector(T *);
	Vector(const Vector &);
	Vector(const Matrix <T> &);

	Vector(const std::vector <T> &);
	Vector(const std::initializer_list <T> &);

	Vector(size_t, T *);
	Vector(size_t = 0, T = T());
	
	Vector(size_t, std::function <T (size_t)>);
	Vector(size_t, std::function <T *(size_t)>);

	const Vector &operator=(const Matrix <T> &);

	size_t size() const;

	T &operator[](size_t);
	const T &operator[](size_t) const;

	// Arithmetic
	void operator+=(const Vector &);
	void operator-=(const Vector &);

	// Conversion operators
	explicit operator int() const;
	explicit operator double() const;

	explicit operator Rational <int> () const;

	explicit operator Complex <double> () const;

	explicit operator Vector <double> () const;
	explicit operator Vector <Rational <int>> () const;

	// Concatenating vectors
	Vector append_above(const T &);
	Vector append_below(const T &);

	T norm() const;

	// Normalization
	void normalize();

	Vector normalized();

	// Non-member functions
	template <class U>
	friend Vector <U> operator*(const Vector <U> &, const U &);

	template <class U>
	friend U inner(const Vector <U> &, const Vector <U> &);

	template <class U>
	friend Vector <U> cross(const Vector <U> &, const Vector <U> &);
};

template <class T>
Vector <T> ::Vector(T *ref)
{
}

template <class T>
Vector <T> ::Vector(const Vector &other) : Matrix <T> (other) {}

template <class T>
Vector <T> ::Vector(const Matrix <T> &other)
{
	*this = other;
}

template <class T>
Vector <T> ::Vector(const std::vector <T> &ref) : Matrix <T> (ref) {}

template <class T>
Vector <T> ::Vector(const std::initializer_list <T> &ref)
	: Vector(std::vector <T> (ref)) {}

template <class T>
Vector <T> ::Vector(size_t rs, T *ref)
{

}

template <class T>
Vector <T> ::Vector(size_t rs, T def) : Matrix <T> (rs, 1, def) {}

template <class T>
Vector <T> ::Vector(size_t rs, std::function <T (size_t)> gen)
	: Matrix <T> (rs, 1, gen) {}

template <class T>
Vector <T> ::Vector(size_t rs, std::function <T *(size_t)> gen)
	: Matrix <T> (rs, 1, gen) {}

template <class T>
const Vector <T> &Vector <T> ::operator=(const Matrix <T> &other)
{
	// Clean this function with native
	// member functions
	if (this != &other) {
		// Member function
		for (size_t i = 0; i < this->rows; i++)
			delete this->m_array[i];

		delete[] this->m_array;

		// Rehabitate
		this->rows = other.get_rows();

		// Allocate
		this->m_array = new T *[this->rows];

		for (size_t i = 0; i < this->rows; i++) {
			this->m_array[i] = new T[1];

			this->m_array[i][0] = other[i][0];
		}
	}

	return *this;
}

template <class T>
size_t Vector <T> ::size() const
{
	return this->rows;
}

template <class T>
T &Vector <T> ::operator[](size_t i)
{
	return this->m_array[i][0];
}

template <class T>
const T &Vector <T> ::operator[](size_t i) const
{
	return this->m_array[i][0];
}

// Move these to matrix
template <class T>
Vector <T> ::operator double() const
{
	return (double) (*this)[0];
}

template <class T>
Vector <T> ::operator int() const
{
	return (int) (*this)[0];
}

template <class T>
void Vector <T> ::operator+=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->rows; i++)
		this->m_array[i][0] += a[i];
}

template <class T>
void Vector <T> ::operator-=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->rows; i++)
		this->m_array[i][0] -= a[i];
}

template <class T>
Vector <T> ::operator Rational <int> () const
{
	return Rational <int> {(*this)[0], 1};
}

template <class T>
Vector <T> ::operator Complex <double> () const
{
	return Complex <double> {(*this)[0]};
}

template <class T>
Vector <T> ::operator Vector <double> () const
{
	std::vector <double> vec;

	for (size_t i = 0; i < size(); i++)
		vec.push_back((*this)[i]);
	
	return Vector <double> {vec};
}

template <class T>
Vector <T> ::operator Vector <Rational <int>> () const
{
	std::vector <Rational <int>> vec;

	for (size_t i = 0; i < size(); i++)
		vec.push_back((*this)[i]);
	
	return Vector <Rational <int>> {vec};
}

// end conversions

template <class T>
Vector <T> Vector <T> ::append_above(const T &x)
{
	size_t t_sz = size();

	std::vector <T> total {x};

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	return Vector(total);
}

template <class T>
Vector <T> Vector <T> ::append_below(const T &x)
{
	size_t t_sz = size();

	std::vector <T> total;

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	total.push_back(x);

	return Vector(total);
}

template <class T>
T Vector <T> ::norm() const
{
	return sqrt(inner(*this, *this));
}

template <class T>
void Vector <T> ::normalize()
{
	T dt = norm();

	for (size_t i = 0; i < size(); i++)
		(*this)[i] /= dt;
}

template <class T>
Vector <T> Vector <T> ::normalized()
{
	std::vector <T> out;

	T dt = norm();

	for (size_t i = 0; i < size(); i++)
		out.push_back((*this)[i]/dt);

	return Vector(out);
}

// Non-member functions
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
		a[i] *= b;
}

template <class T>
T inner(const Vector <T> &a, const Vector <T> &b)
{
	T acc = 0;

	assert(a.size() == b.size());
	for (size_t i = 0; i < a.rows; i++)
		acc += a[i] * b[i];

	return acc;
}

template <class T>
T cross(const Vector <T> &a, const Vector <T> &b)
{
	assert(a.size() == b.size() == 3);

	return {
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]
	};
}

#endif
