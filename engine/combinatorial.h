#ifndef COMBINATORIAL_H_
#define COMBINATORIAL_H_

#include <cmath>

namespace utility {

	template <class T>
	T falling_power(T n, T k)
	{
		T val;
		T i;

		val = 1;

		i = 0;
		while (i < k) {
			n -= 1;
			val *= n;
			i++;
		}

		return val;
	}

	template <class T>
	T binom(T n, T k, T (*gamma)(T) = std::tgamma)
	{
		return falling_power(n, k) / gamma(k);
	}

	template <class T>
	std::vector <T> inverse_bernoulli_sequence(T n)
	{
	}

}

#endif
