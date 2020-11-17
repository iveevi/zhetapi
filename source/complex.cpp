#include <complex.hpp>

namespace zhetapi {

	bool operator<(const Complex <long double> &a, const Complex <long double> &b)
	{
		return norm(a) < norm(b);
	}

	bool operator<=(const Complex <long double> &a, const Complex <long double> &b)
	{
		return norm(a) <= norm(b);
	}

}