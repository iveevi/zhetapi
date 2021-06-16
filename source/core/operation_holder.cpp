#include "../../engine/core/operation_holder.hpp"

namespace zhetapi {

std::string strcodes[] = {
	"add",
	"subtract",
	"multiply",
	"divide",
	"shur",
	"factorial",
	"power",
	"dot",
	"mod",
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
	"lg",
	"equals",
	"neq",
	"greater than",
	"less than",
	"geq",
	"leq",
	"post incr",
	"post decr",
	"pre incr",
	"pre decr",
	"attribute/method",
	"indexing",
	"boolean or",
	"boolean and",
	"absolute value",
	"square root",
	"round",
	"floor",
	"ceil",
	"int"
};

// TODO: avoid overhead by using integer input (straight from the parser)
operation_holder::operation_holder(const std::string &str) : rep(str)
{
	// Store information into some sort of structure
	// TODO: store in a STATIC table, reduces runtime overhead
	// TODO: reorder (rn its based on history)
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
	else if (str == "@")
		code = dot;
	else if (str == "%")
		code = mod;
	else if (str == "sin")
		code = sxn;
	else if (str == "cos")
		code = cxs;
	else if (str == "tan")
		code = txn;
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
	else if (str == "==")
		code = xeq;
	else if (str == "!=")
		code = neq;
	else if (str == ">")
		code = xge;
	else if (str == "<")
		code = xle;
	else if (str == ">=")
		code = geq;
	else if (str == "<=")
		code = leq;
	else if (str == "p++")
		code = pin;
	else if (str == "p--")
		code = pde;
	else if (str == "r++")
		code = rin;
	else if (str == "r--")
		code = rde;
	else if (str == ".")
		code = attribute;
	else if (str == "[]")
		code = indexing;
	else if (str == "||")
		code = bool_or;
	else if (str == "&&")
		code = bool_and;
	else if (str == "abs")
		code = abs_val;
	else if (str == "sqrt")
		code = square_root;
	else if (str == "round")
		code = round_int;
	else if (str == "floor")
		code = floor_int;
	else if (str == "ceil")
		code = ceil_int;
	else if (str == "int")
		code = cast_int;
	else
		throw bad_operation();
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

::std::string operation_holder::dbg_str() const
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
