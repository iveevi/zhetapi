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

namespace core {

namespace primitive {

// TODO: macro-fy, and make inline

// Addition
mk_optn(add, int, int, v1.data.i + v2.data.i);
mk_optn(add, int, double, (long double) v1.data.i + v2.data.d);
mk_optn(add, double, int, v1.data.d + (long double) v2.data.i);
mk_optn(add, double, double, v1.data.d + v2.data.d);

// Subtraction
mk_optn(sub, int, int, v1.data.i - v2.data.i);
mk_optn(sub, int, double, (long double) v1.data.i - v2.data.d);
mk_optn(sub, double, int, v1.data.d - (long double) v2.data.i);
mk_optn(sub, double, double, v1.data.d - v2.data.d);

// Multiplication
mk_optn(mul, int, int, v1.data.i * v2.data.i);
mk_optn(mul, int, double, (long double) v1.data.i * v2.data.d);
mk_optn(mul, double, int, v1.data.d * (long double) v2.data.i);
mk_optn(mul, double, double, v1.data.d * v2.data.d);

// Division
mk_optn(div, int, double, (long double) v1.data.i / v2.data.d);
mk_optn(div, double, int, v1.data.d / (long double) v2.data.i);
mk_optn(div, double, double, v1.data.d / v2.data.d);

Primitive optn_div_int_int_(const Primitive &v1, const Primitive &v2)
{
	long long int x1 = v1.data.i;
	long long int x2 = v2.data.i;

	if (x1 % x2 == 0)
		return Primitive(x1/x2);

	return Primitive((long double) x1/x2);
}

// Filling out the operation base
const VOverload operations[] {
	{},
	{},
	{
		{ovlid(id_int, id_int), &gt_optn(add, int, int)},
		{ovlid(id_int, id_double), &gt_optn(add, int, double)},
		{ovlid(id_double, id_int), &gt_optn(add, double, int)},
		{ovlid(id_double, id_double), &gt_optn(add, double, double)}
	},
	{
		{ovlid(id_int, id_int), &gt_optn(sub, int, int)},
		{ovlid(id_int, id_double), &gt_optn(sub, int, double)},
		{ovlid(id_double, id_int), &gt_optn(sub, double, int)},
		{ovlid(id_double, id_double), &gt_optn(sub, double, double)}
	},
	{
		{ovlid(id_int, id_int), &gt_optn(mul, int, int)},
		{ovlid(id_int, id_double), &gt_optn(mul, int, double)},
		{ovlid(id_double, id_int), &gt_optn(mul, double, int)},
		{ovlid(id_double, id_double), &gt_optn(mul, double, double)}
	},
	{
		{ovlid(id_int, id_double), &gt_optn(div, int, double)},
		{ovlid(id_double, id_int), &gt_optn(div, double, int)},
		{ovlid(id_double, id_double), &gt_optn(div, double, double)},
		{ovlid(id_int, id_int), &gt_optn(div, int, int)}
	}
};

}

}

}
