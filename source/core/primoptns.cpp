#include "../../engine/core/primoptns.hpp"

#include <iostream>

namespace zhetapi {

// TODO: macro-fy, and make inline
Primitive optn_add_int_int(const Primitive &v1, const Primitive &v2)
{
	return Primitive(v1.data.i + v2.data.i);
}

Primitive optn_sub_int_int(const Primitive &v1, const Primitive &v2)
{
	return Primitive(v1.data.i - v2.data.i);
}

Primitive optn_mul_int_int(const Primitive &v1, const Primitive &v2)
{
	return Primitive(v1.data.i * v2.data.i);
}

Primitive optn_div_int_int(const Primitive &v1, const Primitive &v2)
{
	long long int x1 = v1.data.i;
	long long int x2 = v2.data.i;

	if (x1 % x2 == 0)
		return Primitive(x1/x2);

	return Primitive((long double) x1/x2);
}

// Filling out the operation base
const ovlbase opbase[] {
	{
		{ovlid(id_int, id_int), &optn_add_int_int}
	},
	{
		{ovlid(id_int, id_int), &optn_sub_int_int}
	},
	{
		{ovlid(id_int, id_int), &optn_mul_int_int}
	},
	{
		{ovlid(id_int, id_int), &optn_div_int_int}
	}
};

}
