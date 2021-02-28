#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

// C/C++ headers
#include <initializer_list>
#include <iostream>
#include <vector>
#include <cmath>

namespace zhetapi {
		
/**
* @brief Represents a Polynomial, with respect to some single variable x:
*
* \f$P(x) = \sum\limits^n_{k = 0}a_kx^k = a_0 + a_1x + a_2x^2 + \dots\f$
*/
template <class T>
class Polynomial {
	std::vector <T> coeffs;		// Represents the coefficients of
					// the polynomial
public:
	// Constructor
	explicit Polynomial(const std::vector <T> &);
	explicit Polynomial(const std::initializer_list <T> &);

	// Getters
	size_t degree() const;

	T coefficient(size_t) const;

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

	// Output Methods
	template <class U>
	friend std::ostream &operator<<(std::ostream &, const Polynomial <U> &);
};

/**
 * @brief Construct a polyomial from an array of coefficients. The degree of the
 * resulting polynomial is \f$n - 1,\f$ where \f$n\f$ is the size of the array.
 *
 * @param ref The array of coefficients.
 */
template <class T>
Polynomial <T> ::Polynomial(const std::vector <T> &ref) : coeffs(ref)
{
	if (coeffs.size() == 0)
		coeffs = {0};
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

/**
 * @brief Get the degree of the polynomial.
 *
 * @return The degree of the polynomial.
 */
template <class T>
size_t Polynomial <T> ::degree() const
{
	return coeffs.size() - 1;
}

/**
 * @brief Returns \f$a_n,\f$ where \f$a_nx^n\f$ is the \f$n\f$th degree term.
 *
 * @param deg The value \f$n.\f$
 *
 * @return The value \f$a_n\f$
 */
template <class T>
T Polynomial <T> ::coefficient(size_t deg) const
{
	return coeffs[deg];
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

	for (size_t i = 0; i < coeffs.size() - 1; i++)
		out.push_back((coeffs.size() - (i + 1)) * coeffs[i]);

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

	for (size_t i = 0; i < coeffs.size(); i++)
		out.push_back(coeffs[i] / T(coeffs.size() - i));

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
	std::vector <T> qs {coeffs[0]};

	T rem = coeffs[0];
	for (size_t i = 1; i < coeffs.size(); i++) {
		if (i < coeffs.size() - 1)
			qs.push_back(coeffs[i] + root * rem);

		rem = coeffs[i] + rem * root;
	}

	return {Polynomial(qs), rem};
}

template <class T>
T Polynomial <T> ::evaluate(const T &in) const
{
	T acc = 0;

	for (auto c : coeffs)
		acc = in * acc + c;

	return acc;
}

template <class T>
T Polynomial <T> ::operator()(const T &in) const
{
	T acc = 0;

	for (auto c : coeffs)
		acc = in * acc + c;

	return acc;
}

/**
 * @brief Operator for printing the polynomial nicely.
 */
template <class T>
std::ostream &operator<<(std::ostream &os, const Polynomial <T> &p)
{
	if (p.coeffs[0]) {
		if (p.coeffs[0] != 1)
			os << p.coeffs[0];

		if (p.degree() > 0)
			os << "x";

		if (p.degree() > 1)
			os << "^" << p.degree();
	}

	size_t i = 1;
	while (i <= p.degree()) {
		T c = p.coeffs[i];

		if (c == 0) {
			i++;
			continue;
		}

		os << " + ";
		if (c != 1)
			os << c;

		if (p.degree() - i > 0)
			os << "x";

		if (p.degree() - i > 1)
			os << "^" << (p.degree() - i);

		i++;
	}
	
	return os;
}

}

#endif
