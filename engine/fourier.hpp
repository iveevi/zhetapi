#ifndef FOURIER_H_
#define FOURIER_H_

// Engine headers
#include <vector.hpp>

namespace zhetapi {

template <class T>
class FourierSeries {
	Vector <T>	__a;	// Cosine coefficients
	Vector <T>	__b;	// Sine coefficients

	size_t		__asize;
	size_t		__bsize;
public:
	FourierSeries(const Vector <T> &);
	FourierSeries(const Vector <T> &, const Vector <T> &);

	T evaluate(const T &) const;
	T operator()(const T &) const;
};

// TODO: Fix indices
template <class T>
FourierSeries <T> ::FourierSeries(const Vector <T> &coeffs)
{
	size_t n = coeffs.size();

	__bsize = n/2;
	__asize = n - __bsize;

	size_t i = 0;

	if (n % 2)
		__a[i] = coeffs[i++];
	
	while(i < n) {
		__a[i/2] = coeffs[i];
		__b[i/2] = ceoffs[i + 1];

		i++;
	}
}

template <class T>
FourierSeries <T> ::FourierSeries(const Vector <T> &a, const Vector <T> &b)
		: __a(a), __b(b), __asize(a.size()), __bsize(b.size()) {}

template <class T>
T FourierSeries <T> ::evaluate(const T &x) const
{
	// Make more efficient construction kernels for vectors
	// (maybe dont even create a vector: a custom kernel
	// for modified inner products)
	Vector <T> k_cos(__asize,
		[&](size_t i) {
			if (i == 0)
				return T(0.5);
			
			return std::cos(i * x);
		}
	);

	Vector <T> k_sin(__bsize,
		[&](size_t i) {
			return std::cos((i + 1) * x);
		}
	);

	return inner(__a, k_cos) + inner(__b, k_sin);
}

template <class T>
T FourierSeries <T> ::operator()(const T &x) const
{
	return evaluate(x);
}

}

#endif