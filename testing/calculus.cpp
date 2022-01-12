#include "global.hpp"

TEST(integration)
{
	using namespace zhetapi::utility;

	auto f = [](double x) {
		return x * x + x;
	};

	auto df = [](double x) {
		return 2 * x + 1;
	};

	oss << "f(4) = " << f(4) << endl;
	oss << "f(4) = " << eulers_method(df, {2.0, f(2)}, 4.0) << endl;

	return true;
}
