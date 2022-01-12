#include "global.hpp"

TEST(interval_construction)
{
	using namespace zhetapi::utility;

	Interval <> i(5, 10);

	oss << i << endl;

	i = 100.0_I;

	oss << i << endl;

	i = 50_I;

	oss << i << endl;

	i = 1_I;

	oss << i << endl;

	i = 0_I;

	oss << i << endl;

	i = Interval <> ();

	oss << i << endl;

	return true;
}

TEST(interval_sampling)
{
	using namespace zhetapi::utility;
	
	Interval <> i = 100_I;

	for (size_t k = 0; k < 10; k++) {
		long double x = i.uniform();

		oss << "sampled " << x << endl;

		if (x < 0 || x > 100) {
			oss << "\tbad value" << endl;

			return false;
		}
	}

	return true;
}
