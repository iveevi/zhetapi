#include "global.hpp"

TEST(gamma_and_factorial)
{
	using namespace zhetapi::special;

	for (double i = 0; i < 10; i++)
		oss << "ln_gamma(" << (i + 1) << ") = " << ln_gamma(i + 1) << endl;
	
	for (double i = 0; i < 10; i++)
		oss << "ln_factorial(" << i << ") = " << ln_factorial(i) << endl;
	
	try {
		ln_gamma(0);

		return false;
	} catch (const char *err) {
		oss << "\terr: " << err << endl;
	}

	try {
		ln_factorial(-1);

		return false;
	} catch (const char *err) {
		oss << "\terr: " << err << endl;
	}

	return true;
}
