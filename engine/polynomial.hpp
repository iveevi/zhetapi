#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

// C/C++ headers
#include <initializer_list>
#include <iostream>
#include <vector>
#include <cmath>

// Engine headers
#include <vector.hpp>

namespace zhetapi {

// Next power of 2
size_t npow2(size_t);

/**
* @brief Represents a Polynomial, with respect to some single variable x:
*
* \f$P(x) = \sum\limits^n_{k = 0}a_kx^k = a_0 + a_1x + a_2x^2 + \dots\f$
*/
template <class T>
class Polynomial {
protected:
	T *	__coeffs	= nullptr;
	size_t	__degree	= 0;
	size_t	__size		= 0;

	template <class C>
	void indexed_constructor(const C &, size_t);

	void clear();
public:
	// Constructors
	Polynomial();
	Polynomial(const Polynomial &);
	explicit Polynomial(const Vector <T> &);
	explicit Polynomial(const std::vector <T> &);
	Polynomial(const std::initializer_list <T> &);
	Polynomial(T *, size_t);

	Polynomial &operator=(const Polynomial &);

	~Polynomial();

	// Getters
	size_t degree() const;

	T &operator[](size_t);
	const T &operator[](size_t) const;

	// Functional Methods
	Polynomial integrate() const;
	T integrate(const T &, const T &) const;

	Polynomial differentiate() const;
	T differentiate(const T &) const;

	std::vector <T> roots() const;
	std::vector <T> roots(size_t, const T &, const T & = exp(1.0)) const;

	std::pair <Polynomial, T> synthetic_divide(const T &) const;

	T evaluate(const T &) const;
	T operator()(const T &) const;

	// Operations
	template <class U>
	friend Polynomial <U> operator+(
			const Polynomial <U> &,
			const Polynomial <U> &);
	
	template <class U>
	friend Polynomial <U> operator-(
			const Polynomial <U> &,
			const Polynomial <U> &);
	
	template <class U>
	friend Polynomial <U> operator*(
			const Polynomial <U> &,
			const Polynomial <U> &);

	// Comparison
	template <class U>
	friend bool operator==(
			const Polynomial <U> &,
			const Polynomial <U> &);
	
	template <class U>
	friend bool operator!=(
			const Polynomial <U> &,
			const Polynomial <U> &);
};

template <class T>
Polynomial <T> ::Polynomial() {}

template <class T>
Polynomial <T> ::Polynomial(const Polynomial <T> &other)
{
	indexed_constructor(other, other.__degree + 1);
}

template <class T>
Polynomial <T> ::Polynomial(const Vector <T> &ref)
{
	indexed_constructor(ref, ref.size());
}

/**
 * @brief Construct a polyomial from an array of coefficients. The degree of the
 * resulting polynomial is \f$n - 1,\f$ where \f$n\f$ is the size of the array.
 *
 * @param ref The array of coefficients.
 */
template <class T>
Polynomial <T> ::Polynomial(const std::vector <T> &ref)
{
	indexed_constructor(ref, ref.size());
}

/**
 * @brief Construct a polyomial from an initializer list, containng the
 * coefficients of the polynomial.
 *
 * @param ref The initializer list of coefficients.
 */
template <class T>
Polynomial <T> ::Polynomial(const std::initializer_list <T> &ref)
	: Polynomial(std::vector <T> {ref}) {}

template <class T>
Polynomial <T> ::Polynomial(T *coeffs, size_t size)
{
	__size = npow2(size);
	__degree = __size - 1;

	// Walk down to the last non-zero
	while (__degree >= 0 && !coeffs[__degree])
		__degree--;
	
	__coeffs = new T[__size];

	memcpy(__coeffs, coeffs, sizeof(T) * __size);
}

template <class T>
Polynomial <T> &Polynomial <T> ::operator=(const Polynomial <T> &other)
{
	if (this != &other) {
		clear();

		indexed_constructor(other, other.__degree + 1);
	}

	return *this;
}

template <class T>
Polynomial <T> ::~Polynomial()
{
	clear();
}

template <class T>
template <class C>
void Polynomial <T> ::indexed_constructor(const C &container, size_t size)
{
	__degree = size - 1;

	if (__degree < 0)
		return;

	__size = npow2(__degree + 1);
	__coeffs = new T[__size];

	memset(__coeffs, 0, sizeof(T) * __size);
	for (size_t i = 0; i <= __degree; i++)
		__coeffs[i] = container[i];
}

template <class T>
void Polynomial <T> ::clear()
{
	if (__coeffs) {
		delete[] __coeffs;

		__coeffs = nullptr;
	}

	__size = 0;
	__degree = 0;
}

/**
 * @brief Get the degree of the polynomial.
 *
 * @return The degree of the polynomial.
 */
template <class T>
size_t Polynomial <T> ::degree() const
{
	return __degree;
}

template <class T>
T &Polynomial <T> ::operator[](size_t i)
{
	return __coeffs[i];
}

/**
 * @brief Returns \f$a_n,\f$ where \f$a_nx^n\f$ is the \f$n\f$th degree term.
 *
 * @param deg The value \f$n.\f$
 *
 * @return The value \f$a_n\f$
 */
template <class T>
const T &Polynomial <T> ::operator[](size_t i) const
{
	return __coeffs[i];
}

/**
 * @brief Return \f$P',\f$ where \f$P\f$ is the polynomial represented by
 * current object.
 *
 * @return The polynomial \f$P'.\f$
 */
template <class T>
Polynomial <T> Polynomial <T> ::differentiate() const
{
	std::vector <T> out;

	for (size_t i = 0; i < __size - 1; i++)
		out.push_back((__size - (i + 1)) * __coeffs[i]);

	return Polynomial(out);
}

template <class T>
T Polynomial <T> ::differentiate(const T &val) const
{
	return differentiate()(val);
}

/**
 * @brief Integrates the Polynomial, with
 * the constant C being 0.
 */
template <class T>
Polynomial <T> Polynomial <T> ::integrate() const
{
	std::vector <T> out;

	for (size_t i = 0; i < __size; i++)
		out.push_back(__coeffs[i] / T(__size - i));

	out.push_back(0);

	return Polynomial(out);
}

template <class T>
T Polynomial <T> ::integrate(const T &a, const T &b) const
{
	Polynomial prim = integrate();

	return prim(b) - prim(a);
}

template <class T>
std::vector <T> Polynomial <T> ::roots() const
{
	switch (degree()) {
		// TODO: Fill this part out later
	};
}

/**
 * @brief Solves the roots of the representative
 * Polynomial using the Durand-Kerner method.
 *
 * @param rounds The number of iteration to be
 * performed by the method.
 *
 * @param eps The precision threshold; when the
 * sum of the squared difference between the roots
 * of successive iterations is below eps, the method
 * will exit early.
 */
template <class T>
std::vector <T> Polynomial <T> ::roots(
		size_t rounds,
		const T &eps,
		const T &seed) const
{
	std::vector <T> rts;

	T val = 1;
	for (size_t i = 0; i < degree(); i++, val *= seed)
		rts.push_back(val);

	for (size_t i = 0; i < rounds; i++) {
		std::vector <T> nrts(degree());

		for (size_t j = 0; j < rts.size(); j++) {
			T prod = 1;

			for (size_t k = 0; k < rts.size(); k++) {
				if (k != j)
					prod *= rts[j] - rts[k];
			}

			nrts[j] = rts[j] - evaluate(rts[j])/prod;
		}

		T err = 0;

		for (size_t j = 0; j < rts.size(); j++)
			err += (nrts[j] - rts[j]) * (nrts[j] - rts[j]);

		if (err < eps)
			break;

		rts = nrts;
	}

	return rts;
}

template <class T>
std::pair <Polynomial <T>, T> Polynomial <T> ::synthetic_divide(const T &root) const
{
	std::vector <T> qs {__coeffs[0]};

	T rem = __coeffs[0];
	for (size_t i = 1; i < __size; i++) {
		if (i < __size - 1)
			qs.push_back(__coeffs[i] + root * rem);

		rem = __coeffs[i] + rem * root;
	}

	return {Polynomial(qs), rem};
}

template <class T>
T Polynomial <T> ::evaluate(const T &in) const
{
	T acc = 0;

	for (size_t i = 0; i <= __degree; i++)
		acc = in * acc + __coeffs[i];

	return acc;
}

template <class T>
T Polynomial <T> ::operator()(const T &in) const
{
	return evaluate(in);
}

template <class T>
bool operator==(const Polynomial <T> &f, const Polynomial <T> &g)
{
	if (f.__degree != g.__degree)
		return false;

	for (size_t i = 0; i <= f.__degree; i++) {
		if (f[i] != g[i])
			return false;
	}

	return true;
}

template <class T>
bool operator!=(const Polynomial <T> &f, const Polynomial <T> &g)
{
	return !(f == g);
}

// Arithmetic
template <class T>
Polynomial <T> operator+(const Polynomial <T> &f, const Polynomial <T> &g)
{
	const T *ptr = f.__coeffs;
	const T *optr = g.__coeffs;

	size_t size = f.__size;
	size_t ldeg = f.__degree;

	if (g.__size > size) {
		size = g.__size;
		ldeg = g.__degree;

		std::swap(ptr, optr);
	}

	T *coeffs = new T[size];

	memcpy(coeffs, ptr, size * sizeof(T));
	for (size_t i = 0; i <= ldeg; i++)
		coeffs[i] += optr[i];

	Polynomial <T> out(coeffs, size);

	delete[] coeffs;

	return out;
}

template <class T>
Polynomial <T> operator-(const Polynomial <T> &f, const Polynomial <T> &g)
{
	const T *ptr = f.__coeffs;
	const T *optr = g.__coeffs;

	size_t size = f.__size;
	size_t ldeg = f.__degree;

	T sign = 1;
	if (g.__size > size) {
		size = g.__size;
		ldeg = g.__degree;

		std::swap(ptr, optr);

		sign = -1;
	}

	T *coeffs = new T[size];

	memcpy(coeffs, ptr, size * sizeof(T));
	for (size_t i = 0; i <= ldeg; i++) {
		coeffs[i] -= optr[i];

		coeffs[i] *= sign;
	}

	Polynomial <T> out(coeffs, size);

	delete[] coeffs;

	return out;
}

/**
 * @brief Operator for printing the polynomial nicely.
 */
template <class T>
std::ostream &operator<<(std::ostream &os, const Polynomial <T> &p)
{
	size_t degree = p.degree();

	for (size_t i = 0; i <= p.degree(); i++) {
		T x = p[i];

		if (x) {
			os << x;

			if (i > 0)
				os << "x";

			if (i > 1)
				os << "^" << i;

			if (i < degree)
				os << " + ";
		}
	}

	return os;
}

}

#endif
