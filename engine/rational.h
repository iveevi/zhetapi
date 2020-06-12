#ifndef RATIONAL_H_
#define RATIONAL_H_

#include "combinatorial.h"

/**
 * @brief Represents the rational
 * number a/b where a and b are
 * both of type T.
 */
template <class T>
class rational {
public:
	class non_integral_type {};
private:
	T a;
	T b;
public:
	rational(T, T);

	operator double() const;

	rational &operator+=(const rational &);
	rational &operator-=(const rational &);
	rational &operator*=(const rational &);
	rational &operator/=(const rational &);

	template <class U>
	friend std::ostream &operator<<(std::ostream &, const rational <U> &);
private:
	void simplify();
};

//////////////////////////////////////////
// Constructors
//////////////////////////////////////////

template <class T>
rational <T> ::rational(T p, T q) : a(p), b(q)
{
	if (!is_integral <T> ::value)
		throw non_integral_type();

	simplify();
}

//////////////////////////////////////////
// Public Methods
//////////////////////////////////////////

//////////////////////////////////////////
// Operator Overloads
//////////////////////////////////////////
template <class T>
rational <T> ::operator double() const
{
	return (double) a / (double) b;
}

template <class T>
rational <T> &rational <T> ::operator+=(const rational <T> &other)
{
	a = a * other.b + b * other.a;
	b *= other.b;

	simplify();

	return *this;
}

template <class T>
rational <T> &rational <T> ::operator-=(const rational <T> &other)
{
	a = a * other.b - b * other.a;
	b *= other.b;

	simplify();

	return *this;
}

template <class T>
rational <T> &rational <T> ::operator*=(const rational <T> &other)
{
	a *= other.a;
	b *= other.b;

	simplify();

	return *this;
}

template <class T>
rational <T> &rational <T> ::operator/=(const rational <T> &other)
{
	a *= other.b;
	b *= other.a;

	simplify();

	return *this;
}

//////////////////////////////////////////
// I/O Functions
//////////////////////////////////////////

template <class T>
std::ostream &operator<<(std::ostream &os, const rational <T> &rat)
{
	if (rat.a == 0)
		os << 0;
	else if (rat.b == 1)
		os << rat.a;
	else
		os << rat.a << "/" << rat.b;

	return os;
}

//////////////////////////////////////////
// Private Methods
//////////////////////////////////////////

template <class T>
void rational <T> ::simplify()
{
	T tmp = utility::integral_gcd(a, b);

	a /= tmp;
	b /= tmp;
}

#endif
