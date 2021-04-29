#include "math.hpp"

ZHETAPI_REGISTER(zhp_complex)
{
	// Fill out later
	return nullptr;
}

ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT_SYMBOL(complex, zhp_complex);

	ZHETAPI_EXPORT_CONSTANT(e, double, exp(1));
	ZHETAPI_EXPORT_CONSTANT(pi, double, acos(-1));
}
