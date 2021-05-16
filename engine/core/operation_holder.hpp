#ifndef OPERATION_HOLDER_H_
#define OPERATION_HOLDER_H_

// C/C++ headers
#include <string>

// Engine headers
#include <token.hpp>

namespace zhetapi {

enum codes {
	add,
	sub,
	mul,
	dvs,
	shr,
	fct,	// Factorial
	pwr,
	dot,	// Dot product
	mod,	// Modulo operator
	sxn,
	cxs,
	txn,
	csc,
	sec,
	cot,
	snh,
	csh,
	tnh,
	cch,
	sch,
	cth,
	lxg,	// Log base 10
	xln,	// Natural log
	xlg,	// Log base 2
	xeq,	// Equals
	neq,	// Not equals
	xge,	// Greater than
	xle,	// Less than
	geq,	// Greater than or equal to
	leq,	// Less than or equal to
	pin,	// Post increment
	pde,	// Post decrement
	rde,	// Pre incremeny
	rin,	// Pre decrement
	attribute,
	bool_or,
	bool_and,
	abs_val,
	square_root,
	round_int,
	floor_int,
	ceil_int
};

extern std::string strcodes[];

// TODO: turn in to a class
struct operation_holder : public Token {
	std::string rep;

	codes code;

	explicit operation_holder(const std::string &);

	type caller() const override;
	Token *copy() const override;
	std::string dbg_str() const override;
	bool operator==(Token *) const override;

	// Exceptions
	class bad_operation {};
};

}

#endif
