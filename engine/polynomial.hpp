#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

// C/C++ headers
#include <initializer_list>
#include <iostream>
#include <vector>
#include <cmath>

namespace zhetapi {
		
	/**
	* @brief Represents a Polynomial,
	* with respect to the default Variable
	* "x". Should be used for Polynomials
	* to gain a performance boost over
	* the regular Function class objects.
	*/
	template <class T>
	class Polynomial {
		std::vector <T> coeffs;		// Represents the coefficients of
						// the polynomial
	public:
		// Constructor
		Polynomial(const std::vector <T> &);
		Polynomial(const std::initializer_list <T> &);

		// Getters
		size_t degree() const;

		T coefficient(size_t) const;

		// Functional Methods
		Polynomial integrate() const;
		T integrate(const T &, const T &) const;

		Polynomial differentiate() const;
		T differentiate(const T &) const;

		std::vector <T> roots(size_t, const T &, const T & = exp(1.0)) const;

		std::pair <Polynomial, T> synthetic_divide(const T &) const;

		T evaluate(const T &) const;
		T operator()(const T &) const;

		// Output Methods
		template <class U>
		friend std::ostream &operator<<(std::ostream &, const Polynomial <U> &);
	};

	//////////////////////////////////////////
	// Constructors
	//////////////////////////////////////////

	template <class T>
	Polynomial <T> ::Polynomial(const std::vector <T> &ref) : coeffs(ref)
	{
		if (coeffs.size() == 0)
			coeffs = {0};
	}

	template <class T>
	Polynomial <T> ::Polynomial(const std::initializer_list <T> &ref)
		: Polynomial(std::vector <T> {ref}) {}

	//////////////////////////////////////////
	// Getters
	//////////////////////////////////////////

	template <class T>
	size_t Polynomial <T> ::degree() const
	{
		return coeffs.size() - 1;
	}

	template <class T>
	T Polynomial <T> ::coefficient(size_t deg) const
	{
		return coeffs[deg];
	}

	//////////////////////////////////////////
	// Functional Methods
	//////////////////////////////////////////

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
	std::vector <T> Polynomial <T> ::roots(size_t rounds, const T &eps,
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

	//////////////////////////////////////////
	// Output Methods
	//////////////////////////////////////////

	template <class T>
	std::ostream &operator<<(std::ostream &os, const Polynomial <T> &p)
	{
		size_t i = 0;
		if (i >= 0) {
			if (p.coeffs[0]) {
				if (p.coeffs[0] != 1)
					os << p.coeffs[0];

				if (p.degree() > 0)
					os << "x";

				if (p.degree() > 1)
					os << "^" << p.degree();
			}
		}

		i++;
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
