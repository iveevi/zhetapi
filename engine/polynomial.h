#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

#include <initializer_list>
#include <iostream>
#include <vector>

/**
 * @brief Represents a polynomial,
 * with respect to the default variable
 * "x". Should be used for polynomials
 * to gain a performance boost over
 * the regular functor class objects.
 */
template <class T>
class polynomial {
	std::vector <T> coeffs;
public:
	// Constructors
	polynomial(const std::vector <T> &);
	polynomial(const std::initializer_list <T> &);

	// Getters
	size_t degree() const;

	T coefficient(size_t) const;

	// Functional Methods
	polynomial differentiate() const;
	std::pair <polynomial, T> synthetic_divide(const T &) const;

	T operator()(const T &) const;

	// Output Methods
	template <class U>
	friend std::ostream &operator<<(std::ostream &, const polynomial <U> &);
};

template <class T>
polynomial <T> ::polynomial(const std::vector <T> &ref) : coeffs(ref)
{
	if (coeffs.size() == 0)
		coeffs = {0};
}

template <class T>
polynomial <T> ::polynomial(const std::initializer_list <T> &ref)
	: polynomial(std::vector <T> {ref}) {}

template <class T>
size_t polynomial <T> ::degree() const
{
	return coeffs.size() - 1;
}

template <class T>
T polynomial <T> ::coefficient(size_t deg) const
{
	return coeffs[deg];
}

template <class T>
polynomial <T> polynomial <T> ::differentiate() const
{
	std::vector <T> out;

	for (size_t i = 0; i < coeffs.size() - 1; i++)
		out.push_back((coeffs.size() - (i + 1)) * coeffs[i]);

	return polynomial(out);
}

template <class T>
std::pair <polynomial <T>, T> polynomial <T> ::synthetic_divide(const T &root) const
{
	std::vector <T> qs {coeffs[0]};

	T rem = coeffs[0];
	for (size_t i = 1; i < coeffs.size(); i++) {
		if (i < coeffs.size() - 1)
			qs.push_back(coeffs[i] + root * rem);

		rem = coeffs[i] + rem * root;
	}

	return {polynomial(qs), rem};
}

template <class T>
T polynomial <T> ::operator()(const T &in) const
{
	T acc = 0;

	for (auto c : coeffs)
		acc = in * acc + c;

	return acc;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const polynomial <T> &p)
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

#endif
