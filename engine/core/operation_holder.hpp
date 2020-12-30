#ifndef OPERATION_HOLDER_H_
#define OPERATION_HOLDER_H_

// C/C++ headers
#include <string>

// Engine headers
#include <token.hpp>

namespace zhetapi {

	// Make this more flexible
	// to the user later on
	enum codes {
		add,
		sub,
		mul,
		dvs,
		shr,
		fct,	// Factorial
		pwr,
		dot,	// Dot product
		sin,
		cos,
		tan,
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
		xlg	// Log base 2
	};

	extern ::std::string strcodes[];

	struct operation_holder : public Token {
		::std::string rep;

		codes code;

		operation_holder(const ::std::string &);

		type caller() const override;
		Token *copy() const override;
		::std::string str() const override;

		virtual bool operator==(Token *) const override;
	};
	
}

#endif
