#include <std/functions.hpp>

namespace zhetapi {

namespace special {

double ln_gamma(double x)
{
	if (x <= 0)
		throw("ln_gamma: expected a positive argument.");

	static const int N = 14;

	static const double C[] = {
		57.1562356658629235,
		-59.5979603554754912,
		14.1360979747417471,
		-0.491913816097620199,
		0.339946499848118887e-4,
		0.465236289270485756e-4,
		-0.983744753048795646e-4,
		0.158088703224912494e-3,
		-0.210264441724104883e-3,
		0.217439618115212643e-3,
		-0.164318106536763890e-3,
		0.844182239838527433e-4,
		-0.261908384015814087e-4,
		0.368991826595316234e-5
	};

	double tx;
	double ty;
	double tmp;
	double ser;

	ty = tx = x;

	tmp = tx + 5.24218750000000000;
	tmp = (tx + 0.5) * log(tmp) - tmp;
	ser = 0.999999999999997092;

	int i = 0;
	while (i < N)
		ser += C[i++]/(++ty);

	return tmp + log(2.5066282746310005 * ser / tx);
}

double ln_factorial(int x)
{
	static double table[FACTORIAL_BUFFER_SIZE];
	static bool init = true;

	if (init) {
		init = false;

		for (int i = 0; i < FACTORIAL_BUFFER_SIZE; i++)
			table[i] = ln_gamma(i + 1.0);
	}

	if (x < 0)
		throw("ln_factorial: cannot have a negative argument.");

	if (x < FACTORIAL_BUFFER_SIZE)
		return table[x];

	return ln_gamma(x + 1.0);
}

double poisson(double lambda, int k)
{
	double lp = -lambda + k * log(k) - ln_gamma(k + 1);
	return exp(lp);
}

}

}
