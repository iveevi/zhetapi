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

ZHETAPI_REGISTER(zhp_gcd)
{
	// TODO: allow multi param gcd
	
	zhetapi::OpZ *o1, *o2;

	zhetapi::Z val;
	switch (zhetapi::zhetapi_cast_cc(inputs, o1, o2)) {
	case 2:
		val = zhetapi::numtheory::integral_gcd(o1->get(), o2->get());
		break;
	default:
		throw std::runtime_error("gcd: expected 2 integer inputs");
	}

	return new zhetapi::OpZ(val);
}

ZHETAPI_REGISTER(zhp_lcm)
{
	// TODO: allow multi param gcd
	
	zhetapi::OpZ *o1, *o2;

	zhetapi::Z val;
	switch (zhetapi::zhetapi_cast_cc(inputs, o1, o2)) {
	case 2:
		val = zhetapi::numtheory::integral_lcm(o1->get(), o2->get());
		break;
	default:
		throw std::runtime_error("lcm: expected 2 integer inputs");
	}

	return new zhetapi::OpZ(val);
}

ZHETAPI_REGISTER(zhp_modexp)
{
	zhetapi::OpZ *o1, *o2, *o3;

	zhetapi::Z val;
	switch (zhetapi::zhetapi_cast_cc(inputs, o1, o2, o3)) {
	case 3:
		val = zhetapi::numtheory::modexp(o1->get(), o2->get(), o3->get());
		break;
	default:
		throw std::runtime_error("lcm: expected 2 integer inputs");
	}

	return new zhetapi::OpZ(val);
}

ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT_SYMBOL(complex, zhp_complex);
	ZHETAPI_EXPORT_SYMBOL(sieve, zhp_sieve);
	ZHETAPI_EXPORT_SYMBOL(gcd, zhp_gcd);
	ZHETAPI_EXPORT_SYMBOL(lcm, zhp_lcm);
	ZHETAPI_EXPORT_SYMBOL(modexp, zhp_modexp);
	
	ZHETAPI_EXPORT_CONSTANT(e, double, exp(1));
	ZHETAPI_EXPORT_CONSTANT(pi, double, acos(-1));
}
