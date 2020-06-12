#ifndef COMBINATORIAL_H_
#define COMBINATORIAL_H_

#include <cmath>

namespace utility {

	//////////////////////////////////////////
	// Exceptions
	//////////////////////////////////////////

	/* @brief Exception for asserting that
	 * a template parameter must be
	 * of an integral type.
	 */
	class non_integral_type {};

	//////////////////////////////////////////
	// Perliminary Helper Functions
	//////////////////////////////////////////

	/**
	 * @brief Factorial function for integral
	 * types to avoid the usage of the
	 * tgamma library function.
	 */
	template <class T>
	T integral_factorial(T n)
	{
		T val = 1;
		for (T i = 1; i <= n; i++)
			val *= i;

		return val;
	}

	/**
	 * @brief Falling factorial function, available
	 * for all types that support the arithmetic
	 * operations.
	 */
	template <class T>
	T falling_power(T n, T k)
	{
		T val = 1;
		for (T i = 0; i < k; i++)
			val *= (n - i);

		return val;
	}

	//////////////////////////////////////////
	// Binomial Coefficients
	//////////////////////////////////////////
	
	/**
	 * @brief General binomial, computed using the falling
	 * factorial and the gamma function. The gamma
	 * function defaults to the library function
	 * tgamma.
	 *
	 * @param gamma Used gamma function, defaults to std::tgamma
	 */
	template <class T>
	T binom(T n, T k, T (*gamma)(T) = std::tgamma)
	{
		return falling_power(n, k) / gamma(k + 1);
	}

	/**
	 * @brief Integral binomial, utilizes the falling
	 * power function as well as the integral factorial
	 * function.
	 */
	template <class T>
	T integral_binom(T n, T k)
	{
		return falling_power(n, k) / integral_factorial(k);
	}

	//////////////////////////////////////////
	// Binomial Coefficients
	//////////////////////////////////////////

	/**
	 * @brief The Euclidian algorithm for determining
	 * the GCD (Greatest Common Divisor) of two
	 * numbers. Includes overhead from the passed
	 * function.
	 */
	template <class T>
	T gcd(T a, T b, T (*mod)(T, T) = std::fmod, T eps = 0)
	{
		if (a > b)
			std::swap(a, b);

		while (std::abs(mod(b, a)) != 0) {
			b = mod(b, a);

			if (a > b)
				std::swap(a, b);
		}

		return std::min(a, b);
	}

	/**
	 * @brief The LCM (Lowest Commmon Multiple),
	 * algorithm which uses the fact that
	 * (a, b) * [a, b] = ab. Includes overhead
	 * from the modulus function which is passed.
	 */
	template <class T>
	T lcd(T a, T b, T(*mod)(T, T) = std::fmod, T eps = 0)
	{
		return a * b / gcd(a, b, mod, eps);
	}

	
	/**
	 * @brief Integral equivalent of the general
	 * gcd function, preferably used for integer
	 * types.
	 */
	template <class T>
	T integral_gcd(T a, T b)
	{
		if (a > b)
			std::swap(a, b);

		while (b % a != 0) {
			b %= a;

			if (a > b)
				std::swap(a, b);
		}

		return std::min(a, b);
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

	/**
	 * @brief Bernoulli sequence generator, generates
	 * a list (array, vector) of the first arbitrary
	 * number of bernoulli numbers. Uses the general
	 * binomial function.
	 */
	template <class T>
	std::vector <T> bernoulli_sequence_real(T n, T (*gamma)(T) = std::tgamma)
	{
		std::vector <T> ibs = {1};

		T tmp;
		for (T i = 1; i <= n; i++) {
			tmp = 0;

			if (i == 1) {
				ibs.push_back(-0.5);
				continue;
			}

			if (std::fmod(i, 2) == 1) {
				ibs.push_back(0);
				continue;
			}

			for (T j = 0; j < i; j++)
				tmp += binom(i + 1, j, gamma) * ibs[j];
			
			ibs.push_back(-tmp/(i + 1));
		}

		return ibs;
	}

}

#endif
