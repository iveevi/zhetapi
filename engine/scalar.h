#ifndef SCALAR_H_
#define SCALAR_H_

#include <iostream>
#include <type_traits>

#include "number.h"
#include "rational.h"
#include "zcomplex.h"

/**
 * @brief Represents a scalar type,
 * which could be either R (Real),
 * Q (Rational), C_R (Complex with
 * Real and Imaginary parts that
 * are Real) or C_Q (Complex with
 * Real and Imaginary parts that are
 * Rational).
 */
template <class T>
class scalar : public number {
	T dat;
public:
	// Constructor
	scalar(T = T());

	// Conversions
	operator T() const;

	// Operators
	template <class U>
	friend bool operator==(const scalar <U> &, const scalar <U> &);

	// Output
	template <class U>
	friend std::ostream &operator<<(std::ostream &, const scalar <U> &);
};

//////////////////////////////////////////
// Constructors
//////////////////////////////////////////

template <class T>
scalar <T> ::scalar(T x) : dat(x)
{
	if (is_zcomplex_rational <T> ::value)
	 	kind = s_complex_rational;
	else if (is_zcomplex_real <T> ::value)
		kind = s_complex_real;
	else if (is_rational <T> ::value)
		kind = s_rational;
	else if (std::is_integral <T> ::value)
		kind = s_integer;
	else
		kind = s_real;
}

//////////////////////////////////////////
// Conversions
//////////////////////////////////////////

template <class T>
scalar <T> ::operator T() const
{
	return dat;
}

//////////////////////////////////////////
// Operators
//////////////////////////////////////////

template <class T>
bool operator==(const scalar <T> &a, const scalar <T> &b)
{
	return (T) a == (T) b;
}

//////////////////////////////////////////
// Output
//////////////////////////////////////////

template <class T>
std::ostream &operator<<(std::ostream &os, const scalar <T> &c)
{
	os << c.dat << " ["
		<< _sets[c.kind] << "]";

	return os;
}

#endif
