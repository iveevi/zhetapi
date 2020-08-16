#ifndef RATIONAL_H_
#define RATIONAL_H_

#include <iostream>
#include <algorithm>

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
	rational(T = 0, T = 1);

	operator bool() const;
	explicit operator double() const;

	/* Mathematical Operators - Members */
	rational &operator+=(const rational &);
	rational &operator-=(const rational &);
	rational &operator*=(const rational &);
	rational &operator/=(const rational &);

	/* Mathematical Operators - Non-Members */
	template <class U>
	friend rational <U> operator+(const rational <U> &, const rational <U> &);

	template <class U>
	friend rational <U> operator-(const rational <U> &, const rational <U> &);

	template <class U>
	friend rational <U> operator*(const rational <U> &, const rational <U> &);

	template <class U>
	friend rational <U> operator/(const rational <U> &, const rational <U> &);

	/* Boolean Operators - Non Members */
	template <class U>
	friend bool operator==(const rational <U> &, const rational <U> &);
	
	template <class U>
	friend bool operator!=(const rational <U> &, const rational <U> &);
	
	template <class U>
	friend bool operator>(const rational <U> &, const rational <U> &);
	
	template <class U>
	friend bool operator<(const rational <U> &, const rational <U> &);
	
	template <class U>
	friend bool operator>=(const rational <U> &, const rational <U> &);
	
	template <class U>
	friend bool operator<=(const rational <U> &, const rational <U> &);

	/* Output Functions */
	template <class U>
	friend std::ostream &operator<<(std::ostream &, const rational <U> &);
private:
	void simplify();

	static T gcd(T, T);
};

//////////////////////////////////////////
// Constructors
//////////////////////////////////////////

template <class T>
rational <T> ::rational(T p, T q) : a(p), b(q)
{
	if (!std::is_integral <T> ::value)
		throw non_integral_type();

	simplify();
}

//////////////////////////////////////////
// Conversion Operators
//////////////////////////////////////////
template <class T>
rational <T> ::operator double() const
{
	return (double) a / (double) b;
}

template <class T>
rational <T> ::operator bool() const
{
	return a != 0;
}

//////////////////////////////////////////
// Arithmetic Operators
//////////////////////////////////////////

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
	using namespace std;

	cout << "Mutliplication:" << endl;
	cout << "This: " << *this << endl;
	cout << "Other: " << other << endl;

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

template <class T>
rational <T> operator+(const rational <T> &a, const rational <T> &b)
{
	rational <T> out = a;

	out += b;

	return out;
}

template <class T>
rational <T> operator-(const rational <T> &a, const rational <T> &b)
{
	rational <T> out = a;

	out -= b;

	return out;
}

template <class T>
rational <T> operator*(const rational <T> &a, const rational <T> &b)
{
	rational <T> out = a;

	out *= b;

	return out;
}

template <class T>
rational <T> operator/(const rational <T> &a, const rational <T> &b)
{
	rational <T> out = a;

	out /= b;

	return out;
}

//////////////////////////////////////////
// Boolean Operators
//////////////////////////////////////////

template <class T>
bool operator==(const rational <T> &a, const rational <T> &b)
{
	return (a.a == b.a) && (a.b == b.b);
}

template <class T>
bool operator!=(const rational <T> &a, const rational <T> &b)
{
	return !(a == b);
}

template <class T>
bool operator>(const rational <T> &a, const rational <T> &b)
{
	return (a.a * b.b) > (a.b * b.a);
}

template <class T>
bool operator<(const rational <T> &a, const rational <T> &b)
{
	return (a.a * b.b) < (a.b * b.a);
}

template <class T>
bool operator>=(const rational <T> &a, const rational <T> &b)
{
	return (a == b) || (a > b);
}

template <class T>
bool operator<=(const rational <T> &a, const rational <T> &b)
{
	return (a == b) || (a < b);
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
	if (b < 0) {
		a *= -1;
		b *= -1;
	}

	T tmp = gcd(a, b);

	a /= tmp;
	b /= tmp;
}

template <class T>
T rational <T> ::gcd(T a, T b)
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

#endif
