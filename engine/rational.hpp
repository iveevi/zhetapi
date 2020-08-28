#ifndef RATIONAL_H_
#define RATIONAL_H_

#include <iostream>
#include <algorithm>

/**
 * @brief Represents the Rational
 * number a/b where a and b are
 * both of type T.
 */
template <class T>
class Rational {
public:
	class non_integral_type {};
private:
	T a;
	T b;
public:
	Rational(T = 0, T = 1);

	operator bool() const;
	
	explicit operator double() const;

	bool is_inf() const;

	/* Mathematical Operators - Members */
	Rational &operator+=(const Rational &);
	Rational &operator-=(const Rational &);
	Rational &operator*=(const Rational &);
	Rational &operator/=(const Rational &);

	/* Mathematical Operators - Non-Members */
	template <class U>
	friend Rational <U> operator+(const Rational <U> &, const Rational <U> &);

	template <class U>
	friend Rational <U> operator-(const Rational <U> &, const Rational <U> &);

	template <class U>
	friend Rational <U> operator*(const Rational <U> &, const Rational <U> &);

	template <class U>
	friend Rational <U> operator/(const Rational <U> &, const Rational <U> &);

	/* Boolean Operators - Non Members */
	template <class U>
	friend bool operator==(const Rational <U> &, const Rational <U> &);
	
	template <class U>
	friend bool operator!=(const Rational <U> &, const Rational <U> &);
	
	template <class U>
	friend bool operator>(const Rational <U> &, const Rational <U> &);
	
	template <class U>
	friend bool operator<(const Rational <U> &, const Rational <U> &);
	
	template <class U>
	friend bool operator>=(const Rational <U> &, const Rational <U> &);
	
	template <class U>
	friend bool operator<=(const Rational <U> &, const Rational <U> &);

	template <class U>
	friend Rational <U> abs(const Rational <U> &);

	/* Output Functions */
	template <class U>
	friend std::ostream &operator<<(std::ostream &, const Rational <U> &);
private:
	void simplify();

	static T gcd(T, T);
};

//////////////////////////////////////////
// Constructors
//////////////////////////////////////////

template <class T>
Rational <T> ::Rational(T p, T q) : a(p), b(q)
{
	if (!std::is_integral <T> ::value)
		throw non_integral_type();

	simplify();
}

//////////////////////////////////////////
// Conversion Operators
//////////////////////////////////////////

template <class T>
Rational <T> ::operator bool() const
{
	return a != 0;
}

template <class T>
Rational <T> ::operator double() const
{
	return (double) a / (double) b;
}

template <class T>
bool Rational <T> ::is_inf() const
{
	return b == 0;
}

//////////////////////////////////////////
// Arithmetic Operators
//////////////////////////////////////////

template <class T>
Rational <T> &Rational <T> ::operator+=(const Rational <T> &other)
{
	a = a * other.b + b * other.a;
	b *= other.b;

	simplify();

	return *this;
}

template <class T>
Rational <T> &Rational <T> ::operator-=(const Rational <T> &other)
{
	a = a * other.b - b * other.a;
	b *= other.b;

	simplify();

	return *this;
}

template <class T>
Rational <T> &Rational <T> ::operator*=(const Rational <T> &other)
{
	using namespace std;

	a *= other.a;
	b *= other.b;

	simplify();

	return *this;
}

template <class T>
Rational <T> &Rational <T> ::operator/=(const Rational <T> &other)
{
	a *= other.b;
	b *= other.a;

	simplify();

	return *this;
}

template <class T>
Rational <T> operator+(const Rational <T> &a, const Rational <T> &b)
{
	Rational <T> out = a;

	out += b;

	return out;
}

template <class T>
Rational <T> operator-(const Rational <T> &a, const Rational <T> &b)
{
	Rational <T> out = a;

	out -= b;

	return out;
}

template <class T>
Rational <T> operator*(const Rational <T> &a, const Rational <T> &b)
{
	Rational <T> out = a;

	out *= b;

	return out;
}

template <class T>
Rational <T> operator/(const Rational <T> &a, const Rational <T> &b)
{
	Rational <T> out = a;

	out /= b;

	return out;
}

//////////////////////////////////////////
// Boolean Operators
//////////////////////////////////////////

template <class T>
bool operator==(const Rational <T> &a, const Rational <T> &b)
{
	return (a.a == b.a) && (a.b == b.b);
}

template <class T>
bool operator!=(const Rational <T> &a, const Rational <T> &b)
{
	return !(a == b);
}

template <class T>
bool operator>(const Rational <T> &a, const Rational <T> &b)
{
	return (a.a * b.b) > (a.b * b.a);
}

template <class T>
bool operator<(const Rational <T> &a, const Rational <T> &b)
{
	return (a.a * b.b) < (a.b * b.a);
}

template <class T>
bool operator>=(const Rational <T> &a, const Rational <T> &b)
{
	return (a == b) || (a > b);
}

template <class T>
bool operator<=(const Rational <T> &a, const Rational <T> &b)
{
	return (a == b) || (a < b);
}

//////////////////////////////////////////
// I/O Functions
//////////////////////////////////////////

template <class T>
std::ostream &operator<<(std::ostream &os, const Rational <T> &rat)
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
void Rational <T> ::simplify()
{
	if (b < 0) {
		a *= -1;
		b *= -1;
	}

	T tmp = gcd(a, b);

	a /= tmp;
	b /= tmp;
}

template <class T>
T Rational <T> ::gcd(T a, T b)
{
		if (a == 0 || b == 0)
			return 1;

		a = std::abs(a);
		b = std::abs(b);

		if (a > b)
			std::swap(a, b);

		while (b % a != 0) {
			b %= a;

			if (a > b)
				std::swap(a, b);
		}

		return std::min(a, b);
}

// Extra functions

template <class T>
Rational <T> abs(const Rational <T> &a)
{
	if (a < Rational <int> {0, 1})
		return {-a.a, a.b};
	
	return a;
}

#endif
