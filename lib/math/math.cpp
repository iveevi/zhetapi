#include "math.hpp"

ZHETAPI_REGISTER(zhp_complex)
{
	// TODO: Fill out later
	return nullptr;
}

// TODO: refactor inputs to args
ZHETAPI_REGISTER(zhp_sieve)
{
	// TODO: assert
	assert(inputs.size() == 1);
	
	// TODO: put in numtheorey.hpp

	zhetapi::Targs tprimes;

	zhetapi::OpZ *lim;
	zhetapi_cast(inputs, lim);
	
	std::vector <zhetapi::Z> primes = zhetapi::numtheory::sieve(lim->get());

	for (zhetapi::Z i : primes)
		tprimes.push_back(new zhetapi::OpZ(i));

	return new zhetapi::Collection(tprimes);
}

ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT_SYMBOL(complex, zhp_complex);
	ZHETAPI_EXPORT_SYMBOL(sieve, zhp_sieve);

	ZHETAPI_EXPORT_CONSTANT(e, double, exp(1));
	ZHETAPI_EXPORT_CONSTANT(pi, double, acos(-1));
}
