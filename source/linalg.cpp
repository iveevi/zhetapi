#include <std/linalg.hpp>

namespace zhetapi {

namespace linalg {

// Using the algorithm from https://arxiv.org/abs/1707.05037
Vector <long long int> pslq(const Vector <long double> &a)
{
	// Length of a
	size_t n = a.size();

	// Partial sums
	Vector <long double> s(n,
		[&](size_t j) -> long double {
			long double sum = 0;

			for (size_t k = j; k < n; k++)
				sum += a[k] * a[k];
			
			return sqrt(sum);
		}
	);

	// Construct the matrix H
	Matrix <long double> H(n, n,
		[&](size_t i, size_t j) -> long double {
			if ((i < j) && (j < n - 1))
				return 0;
			else if ((i == j) && (i < n - 1))
				return s[i + 1]/s[i];
			
			return -(a[i] * a[j])/(s[j] * s[j + 1]);
		}
	);

	return {};
}

}

}