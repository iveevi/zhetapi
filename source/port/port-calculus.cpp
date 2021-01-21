#include "port.hpp"

bool integration()
{
	using namespace zhetapi::utility;

	auto f = [](double x) {
		return x * x + x;
	};

	auto df = [](double x) {
		return 2 * x + 1;
	};

	cout << "f(4) = " << f(4) << endl;
	cout << "f(4) = " << eulers_method(df, {2, f(2)}, 4) << endl;

	return true;
}