#ifndef NUMBER_THEORY_H_
#define NUMBER_THEORY_H_

// C/C++ headers
#include <cmath>
#include <vector>

namespace zhetapi {

namespace numtheory {

// TODO: use f convention for gcd/lcm

/**
 * @brief Integral equivalent of the general
 * gcd function, preferably used for integer
 * types.
 */
template <class T>
T integral_gcd(T a, T b)
{
	if (a == 0 || b == 0)
		return 1;

	a = ::std::abs(a);
	b = ::std::abs(b);

	if (a > b)
		::std::swap(a, b);

	while (b % a != 0) {
		b %= a;

		if (a > b)
			::std::swap(a, b);
	}

	return ::std::min(a, b);
}

/**
 * @brief Integral equivalent of the
 * lcm function.
 */
template <class T>
T integral_lcm(T a, T b)
{
	return a * b / integral_gcd(a, b);
}

// TODO: add totient function

template <class T>
T modmul(T a, T b, T mod)
{
	// Checks
	if (a == 0 || b == 0)
		return 0;

	if (a == 1)
		return (b % mod);

	if (b == 1)
		return (a % mod);

	T hmul = modmul(a, b/2, mod);
	if ((b & 1) == 0)
		return (hmul + hmul) % mod;
	else
		return ((a % mod) + (hmul + hmul)) % mod;
}

// Integral only (add f variants)
template <class T>
T modexp(T base, T exp, T mod)
{
	// Add a string to print on failure (throw actually)
	assert(mod > 1);

	if (!exp)
		return 1;
	else if (exp == 1)
		return (base % mod);

	T hexp = exp << 1;
	T tmp = modexp(base, hexp, mod);

	tmp = modmul(tmp, tmp, mod);
	if (exp & 1)
		tmp = modmul(base, tmp, mod);

	return tmp;
}

/* template <class T>
T modexp(T base, T exp, T mod, T totient)
{
	// Add a string to print on failure (throw actually)
	assert(mod > 1);

	if (!exp)
		return 1;
	else if (exp == 1)
		return (base % mod);

	T hexp = exp << 1;
	T tmp = modexp(base, hexp, mod);

	tmp = (tmp * tmp) % mod;
	if (exp & 0x1)
		tmp = (tmp * base) % mod;

	return tmp;
} */

// TODO: change to only integral types
template <class T = long long>
std::vector <T> sieve(T lim)
{
	std::vector <T> primes = {2};
	
	if (lim < 2)
		return {};

	for (T i = 3; i < lim; i++) {
		bool prime = true;
		
		for (size_t j = 0; primes[j] <= sqrt(i); j++) {
			if (i % primes[j] == 0) {
				prime = false;

				break;
			}
		}

		if (prime)
			primes.push_back(i);
	}

	return primes;
}

}

}

#endif
