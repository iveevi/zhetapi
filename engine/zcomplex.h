#ifndef ZCOMPLEX_H_
#define ZCOMPLEX_H_

#include <complex>
#include <initializer_list>
#include <iostream>

/**
 * @brief The zcomplex class is an
 * extension of the std::complex
 * class which includes a more convenient
 * set of methods, such as normalization.
 */
template <class T>
class zcomplex : public std::complex <T> {
public:
	// Constructor
	zcomplex(const T & = 0);
	zcomplex(const T &, const T &);

	zcomplex(const std::complex <T> &);

	// Getters
	T magnitude() const;

	bool is_real() const;

	operator double() const;

	// Functional Methods
	zcomplex normalize() const;

	// Output Methods
	template <class U>
	friend std::string std::to_string(const zcomplex <U> &);

	template <class U>
	friend std::ostream &operator<<(std::ostream &, const zcomplex <U> &);
};

//////////////////////////////////////////
// Constructors
//////////////////////////////////////////

template <class T>
zcomplex <T> ::zcomplex(const T &re)
	: std::complex <T> (re) {}

template <class T>
zcomplex <T> ::zcomplex(const T &re, const T &im)
	: std::complex <T> (re, im) {}

template <class T>
zcomplex <T> ::zcomplex(const std::complex <T> &z)
	: std::complex <T> (z) {}

//////////////////////////////////////////
// Getters
//////////////////////////////////////////

template <class T>
T zcomplex <T> ::magnitude() const
{
	return sqrt(norm(*this));
}

template <class T>
bool zcomplex <T> ::is_real() const
{
	return this->imag() == 0;
}

template <class T>
zcomplex <T> ::operator double() const
{
	return (double) this->real();
}

//////////////////////////////////////////
// Functional Methods
//////////////////////////////////////////

template <class T>
zcomplex <T> zcomplex <T> ::normalize() const
{
	return *this/magnitude();
}

bool operator<(const zcomplex <long double> &a, const zcomplex <long double> &b)
{
	return norm(a) < norm(b);
}

bool operator<=(const zcomplex <long double> &a, const zcomplex <long double> &b)
{
	return norm(a) <= norm(b);
}

//////////////////////////////////////////
// Output Methods
//////////////////////////////////////////
namespace std {

	template <class T>
	std::string to_string(const zcomplex <T> &z)
	{
		std::string str;

		bool pl = false;

		if (z.real()) {
			pl = true;
			str += to_string(z.real());
		}

		if (z.imag()) {
			if (pl)
				str += " + ";
			str += to_string(z.imag()) + "i";
		}

		return str;
	}

}

template <class T>
std::ostream &operator<<(std::ostream &os, const zcomplex <T> &z)
{
	bool pl = false;

	if (!(z.real() || z.imag())) {
		os << "0";
		return os;
	}

	if (z.real()) {
		pl = true;
		os << z.real();
	}

	if (z.imag()) {
		if (pl)
			os << " + ";
		os << z.imag() << "i";
	}

	return os;
}

#endif
