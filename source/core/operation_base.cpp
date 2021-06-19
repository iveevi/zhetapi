#include "../../engine/core/operation_base.hpp"

#define simple_binary(name, type, op)					\
	Token *name(Token *arg1, Token *arg2)				\
	{								\
		return new type(					\
			(dynamic_cast <type *> (arg1))->get()		\
			+ (dynamic_cast <type *> (arg2))->get()		\
		);							\
	}

namespace zhetapi {

// String table
const char *OpStr[] {
	"addition",
	"subtraction",
	"multiplication",
	"division"
};

// Addition
simple_binary(optn_add_zz, OpZ, +);

// Separate later
simple_binary(optn_sub_zz, OpZ, -);
simple_binary(optn_mul_zz, OpZ, *);
simple_binary(optn_div_zz, OpZ, /);

simple_binary(optn_add_qq, OpQ, +);
simple_binary(optn_add_rr, OpR, +);
simple_binary(optn_add_ccz, OpCmpZ, +);

// Up to 256 operations
const OverloadBase OperationBase[] {
	{ // Addition
		{overload_hash <OpZ, OpZ> (), &optn_add_zz},
		{overload_hash <OpQ, OpQ> (), &optn_add_qq},
		{overload_hash <OpR, OpR> (), &optn_add_rr},
		{overload_hash <OpCmpZ, OpCmpZ> (), &optn_add_ccz}
	},
	{ // Subtraction
		{overload_hash <OpZ, OpZ> (), &optn_sub_zz},
	},
	{ // Multiplication
		{overload_hash <OpZ, OpZ> (), &optn_mul_zz},
	},
	{ // Division
		{overload_hash <OpZ, OpZ> (), &optn_div_zz},
	}
};

}
