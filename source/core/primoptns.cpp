#include "../../engine/core/primoptns.hpp"

#include <iostream>

// Macros
#define mk_optn(op, t1, t2, ftr)				\
	inline Primitive optn_##op##_##t1##_##t2##_		\
		(const Primitive &v1, const Primitive &v2)	\
	{							\
		return Primitive(ftr);				\
	}

#define gt_optn(op, t1, t2) optn_##op##_##t1##_##t2##_

namespace zhetapi {

// TODO: macro-fy, and make inline

// Addition
mk_optn(add, int, int, v1.data.i + v2.data.i);
mk_optn(add, int, double, (long double) v1.data.i + v2.data.d);
mk_optn(add, double, int, v1.data.d + (long double) v2.data.i);

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
		{ovlid(id_int, id_int), &gt_optn(add, int, int)},
		{ovlid(id_int, id_double), &gt_optn(add, int, double)},
		{ovlid(id_double, id_int), &gt_optn(add, double, int)}
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
