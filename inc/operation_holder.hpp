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

	std::string strcodes[] = {
		"add",
		"subtract",
		"multiply",
		"divide",
		"shur",
		"factorial",
		"power",
		"dot",
		"sin",
		"cos",
		"tan",
		"csc",
		"sec",
		"cot",
		"sinh",
		"cosh",
		"tanh",
		"csch",
		"sech",
		"coth",
		"log",
		"ln",
		"lg"
	};

	struct operation_holder : public Token {
		std::string rep;

		codes code;

		operation_holder(const std::string &);

		type caller() const override;
		Token *copy() const override;
		std::string str() const override;

		virtual bool operator==(Token *) const override;
	};

	operation_holder::operation_holder(const std::string &str) : rep(str)
	{
		if (str == "+")
			code = add;
		else if (str == "-")
			code = sub;
		else if (str == "*")
			code = mul;
		else if (str == "/")
			code = dvs;
		else if (str == "shur")
			code = shr;
		else if (str == "!")
			code = fct;
		else if (str == "^")
			code = pwr;
		else if (str == ".")
			code = dot;
		else if (str == "sin")
			code = sin;
		else if (str == "cos")
			code = cos;
		else if (str == "tan")
			code = tan;
		else if (str == "csc")
			code = csc;
		else if (str == "sec")
			code = sec;
		else if (str == "cot")
			code = cot;
		else if (str == "sinh")
			code = snh;
		else if (str == "cosh")
			code = csh;
		else if (str == "tanh")
			code = tnh;
		else if (str == "csch")
			code = cch;
		else if (str == "sech")
			code = sch;
		else if (str == "coth")
			code = cth;
		else if (str == "log")
			code = lxg;
		else if (str == "ln")
			code = xln;
		else if (str == "lg")
			code = xlg;

	}

	Token::type operation_holder::caller() const
	{
		return Token::oph;
	}

	Token *operation_holder::copy() const
	{
		// instead of having to re-evaluate
		// codes, pass the code as well
		return new operation_holder(rep);
	}

	std::string operation_holder::str() const
	{
		return rep + " [" + strcodes[code] + "]";
	}

	bool operation_holder::operator==(Token *tptr) const
	{
		operation_holder *oph = dynamic_cast <operation_holder *> (tptr);

		if (oph == nullptr)
			return false;

		return oph->rep == rep;
	}
}

#endif
