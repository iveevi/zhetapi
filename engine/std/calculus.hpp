#ifndef CALCULUS_H_
#define CALCULUS_H_

// C/C++ headers
#include <map>
#include <type_traits>
#include <vector>
#include <cmath>

// Engine headers
#include <function.hpp>
#include <vector.hpp>
#include <parametrization.hpp>

namespace zhetapi {

namespace utility {

// Single variable, scalar integration of f over [a, b]
template <class T, class F>
T sv_integral(F f, T a, T b, size_t partition_size = 1000)
{
	T step = (b - a)/partition_size;

	// Use a more accurate method later (Simpon's rule, etc.)
	T sum = 0;
	for (T x = a; x <= b; x += step)
		sum += f(x) * step;

	return sum;
}

// Euler's method for single variable scalar functions
template <class T, class Df>
T eulers_method(Df df, Vector <T> given, T x, size_t partition_size = 1000)
{
	T step = (x - given[0])/partition_size;

	T stop = x;
	while (fabs(given[0] - stop) > 1e-10) {
		given[1] += df(given[0]) * step;
		given[0] += step;
	}

	return given[1];
}

template <class T, class P, class F>
T line_integral(Parametrization <T, P, F> prm)
{
	// Always start from the beginning
	prm.reset();

	T sum = 0;

	do {
		sum += inner(prm.value(), prm.dpos());
	} while (prm.step());

	return sum;
}

/**
 * @brief Solves the homogenous linear differential
 * equation (with constant coefficients) whose coefficients
 * are represented by the polynomial that is passed into the
 * function.
 *
 * @tparam T Represents the scalar field; the complex roots of
 * the polynomial are AUTOMATICALLY generated from this function.
 *
 * @return out Represents the basis of functions such that a
 * linear combination of the functions is a solution to the homogenous
 * linear differential equation with constant coefficients.
 
template <class T>
std::vector <Function> solve_hlde_constant(const polynomial <Complex <T>> &p,
		size_t rounds = 10000, const Complex <T> &eps = 1E-100L,
		const Complex <T> &start = {0.4, 0.9})
{
	std::vector <Complex <T>> roots = p.roots(rounds, eps, start);

	std::vector <Function <Complex <T>>> out;

	std::vector <Complex <T>> inserted;

	table <Complex <T>> tbl {
		Variable <Complex <T>> {"e", false, exp(1)}
	};

	for (auto vl : roots) {
		if (vl == Complex <T> {0, 0})
			continue;

		auto itr = ::std::find_if(inserted.begin(), inserted.end(), [&](const Complex <T> &a) {
			return pow(norm(vl - a), 10.5) < norm(eps);
		});

		if (itr != inserted.end()) {
			size_t deg = ::std::count_if(inserted.begin(), inserted.end(), [&](const Complex <T> &a) {
				return pow(norm(vl - a), 10.5) < norm(eps);
			});
			
			if (vl.is_real()) {
				out.push_back({"f", {"x"}, "x^" + ::std::to_string(deg) + " * e^("
						+ ::std::to_string(vl.real()) + " * x)", tbl});
			} else {
				out.push_back({"f", {"x"}, "x^" + ::std::to_string(deg) + " * e^("
						+ ::std::to_string(vl.real()) + " * x)" + " * cos("
						+ ::std::to_string(vl.imag()) + " * x)", tbl});
			}
		} else {
			inserted.push_back(vl);

			if (vl.is_real()) {
				out.push_back({"f", {"x"}, "e^(" + ::std::to_string(vl.real()) + " * x)", tbl});
			} else {
				out.push_back({"f", {"x"}, "e^(" + ::std::to_string(vl.real()) + " * x)"
						+ " * cos(" + ::std::to_string(vl.imag()) + " * x)", tbl});
			}
		}
	}

	size_t deg = ::std::count(roots.begin(), roots.end(), Complex <T> {0, 0});

	if (deg > 0)
		out.push_back({"f", {"x"}, "x^" + ::std::to_string(deg - 1), tbl});

	return out;
} */

}

}

#endif
