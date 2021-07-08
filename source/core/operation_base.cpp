#include "../../engine/core/operation_base.hpp"
#include "../../engine/std/combinatorial.hpp"

namespace zhetapi {

namespace detail {

// Functions first (the definitions take up a lot of space)
Token *compute(const std::string &str, const Targs &args)
{
	// Check presence in universal operation sets
	if (universals.find(str) != universals.end())
		return universals[str](args);
	
	if (operations.find(str) == operations.end())
		throw bad_operation(str);
	
	// Generature the signature
	Signature sig = gen_signature(args);
	Overload ovld = operations[str];

	size_t len = sig.size();
	for (const auto &pair : ovld) {
		if (pair.first.size() != sig.size())
			continue;
		
		bool matches = true;
		for (size_t i = 0; i < sig.size(); i++) {
			if (sig[i] != pair.first[i]) {
				matches = false;

				break;
			}
		}

		if (matches)
			return pair.second(args);
	}
	
	throw bad_overload(gen_overload_msg(sig, str));
}

std::string overload_catalog(const std::string &str)
{
	static const size_t linelen = 25;

	std::string out = "Available overloads for \"" + str + "\": {\n";
	std::string tmp;

	Overload ovld = operations[str];
	for (size_t i = 0; i < ovld.size(); i++) {
		Signature sig = ovld[i].first;

		tmp += "(";

		for (size_t j = 0; j < sig.size(); j++) {
			tmp += types::symbol(sig[j]);

			if (j < sig.size() - 1)
				tmp += ", ";
		}

		tmp += ")";
		if (i < ovld[i].first.size() - 1)
			tmp += ", ";
		
		if (tmp.length() > linelen) {
			tmp += "\n";

			out += "\t" + tmp;
			tmp.clear();
		}
	}

	if (!tmp.empty())
		out += "\t" + tmp;
	
	return out + "\n\t}";
}

std::string gen_overload_msg(const Signature &sig, const std::string &str)
{
	return "Unknown overload " + gen_signature_str(sig)
		+ " for operation \"" + str + "\".\n\t"
		+ overload_catalog(str);
}

std::string gen_signature_str(const Signature &sig)
{
	std::string out = "(";

	for (size_t i = 0; i < sig.size(); i++) {
		// TODO: change types to detail
		out += types::symbol(sig[i]);
		
		if (i < sig.size() - 1)
			out += ", ";
	}
	
	return out + ")";
}

Signature gen_signature(const Targs &args)
{
	Signature sig;

	for (const Token *tptr : args) {
		if (!tptr)
			throw bad_signature();
		
		sig.push_back(typeid(*tptr));
	}

	return sig;
}

// Univeral (no overload) operations
std::unordered_map <std::string, Operation> universals {
	{"==",
		[](const Targs &args) {
			return new Operand <bool> (tokcmp(args[0], args[1]));
		}
	},
	{"!=",
		[](const Targs &args) {
			return new Operand <bool> (!tokcmp(args[0], args[1]));
		}
	}
};

// Functions for the operations
long double lg(long double x) {return std::log(x)/std::log(2);}
long double csc(long double x) {return 1.0/std::sin(x);}
long double sec(long double x) {return 1.0/std::cos(x);}
long double cot(long double x) {return 1.0/std::tan(x);}
long double csch(long double x) {return 1.0/std::sinh(x);}
long double sech(long double x) {return 1.0/std::cosh(x);}
long double coth(long double x) {return 1.0/std::tanh(x);}
long double log(long double x, long double y) {return std::log(y)/std::log(x);}

// Operations
std::unordered_map <std::string, Overload> operations {
	{"+", {
		casted_binary_operation_set(+),
		casted_vector_binary_operation_set(+),
		casted_matrix_binary_operation_set(+)
	}},
	{"-", {
		casted_binary_operation_set(-),
		casted_vector_binary_operation_set(-),
		casted_matrix_binary_operation_set(-)
	}},
	{"*", {casted_binary_operation_set(*)}},
	{"/", {
		casted_binary_operation(Z, Z, Q, /),
		casted_binary_operation(Q, Q, Q, /),
		casted_binary_operation(R, R, R, /),
		casted_binary_operation(CmpZ, CmpZ, CmpQ, /),
		casted_binary_operation(CmpQ, CmpQ, CmpQ, /),
		casted_binary_operation(CmpR, CmpR, CmpR, /),
		het_casted_binary_operation(Z, Q, Q, /),
		het_casted_binary_operation(Z, R, R, /),
		het_casted_binary_operation(Z, CmpZ, CmpZ, /),
		het_casted_binary_operation(Z, CmpQ, CmpQ, /),
		het_casted_binary_operation(Z, CmpR, CmpR, /),
		het_casted_binary_operation(Q, R, R, /),
		het_casted_binary_operation(Q, CmpZ, CmpQ, /),
		het_casted_binary_operation(Q, CmpQ, CmpQ, /),
		het_casted_binary_operation(Q, CmpR, CmpR, /),
		het_casted_binary_operation(R, CmpZ, CmpR, /),
		het_casted_binary_operation(R, CmpQ, CmpR, /),
		het_casted_binary_operation(R, CmpR, CmpR, /),
		het_casted_binary_operation(CmpZ, CmpQ, CmpQ, /),
		het_casted_binary_operation(CmpZ, CmpR, CmpR, /),
		het_casted_binary_operation(CmpQ, CmpR, CmpR, /)
	}},
	{"^", {
		binary_operation_ftn(Z, Z, Z, pow),
		binary_operation_ftn(Q, Q, R, pow),
		binary_operation_ftn(R, R, R, pow)
	}},
	{">", {scalar_binary_operation_set(>)}},
	{"<", {scalar_binary_operation_set(<)}},
	{">=", {scalar_binary_operation_set(>=)}},
	{"<=", {scalar_binary_operation_set(<=)}},
	{"p++", {
		unary_operation_blank(Z, Z, a->get() + 1),
		unary_operation_blank(Q, Q, a->get() + 1),
		unary_operation_blank(R, R, a->get() + 1)
	}},
	{"p--", {
		unary_operation_blank(Z, Z, a->get() - 1),
		unary_operation_blank(Q, Q, a->get() - 1),
		unary_operation_blank(R, R, a->get() - 1)
	}},
	{"r++", {
		unary_operation_blank(Z, Z, a->get() + 1),
		unary_operation_blank(Q, Q, a->get() + 1),
		unary_operation_blank(R, R, a->get() + 1)
	}},
	{"r--", {
		unary_operation_blank(Z, Z, a->get() - 1),
		unary_operation_blank(Q, Q, a->get() - 1),
		unary_operation_blank(R, R, a->get() - 1)
	}},
	{"sin", {
		casted_unary_operation_ftn(Z, R, std::sin),
		casted_unary_operation_ftn(Q, R, std::sin),
		casted_unary_operation_ftn(R, R, std::sin)
	}},
	{"cos", {
		casted_unary_operation_ftn(Z, R, std::cos),
		casted_unary_operation_ftn(Q, R, std::cos),
		casted_unary_operation_ftn(R, R, std::cos)
	}},
	{"tan", {
		casted_unary_operation_ftn(Z, R, std::tan),
		casted_unary_operation_ftn(Q, R, std::tan),
		casted_unary_operation_ftn(R, R, std::tan)
	}},
	{"csc", {
		casted_unary_operation_ftn(Z, R, csc),
		casted_unary_operation_ftn(Q, R, csc),
		casted_unary_operation_ftn(R, R, csc)
	}},
	{"sec", {
		casted_unary_operation_ftn(Z, R, sec),
		casted_unary_operation_ftn(Q, R, sec),
		casted_unary_operation_ftn(R, R, sec)
	}},
	{"cot", {
		casted_unary_operation_ftn(Z, R, cot),
		casted_unary_operation_ftn(Q, R, cot),
		casted_unary_operation_ftn(R, R, cot)
	}},
	{"sinh", {
		casted_unary_operation_ftn(Z, R, std::sinh),
		casted_unary_operation_ftn(Q, R, std::sinh),
		casted_unary_operation_ftn(R, R, std::sinh)
	}},
	{"cosh", {
		casted_unary_operation_ftn(Z, R, std::cosh),
		casted_unary_operation_ftn(Q, R, std::cosh),
		casted_unary_operation_ftn(R, R, std::cosh)
	}},
	{"tanh", {
		casted_unary_operation_ftn(Z, R, std::tanh),
		casted_unary_operation_ftn(Q, R, std::tanh),
		casted_unary_operation_ftn(R, R, std::tanh)
	}},
	{"csch", {
		casted_unary_operation_ftn(Z, R, csch),
		casted_unary_operation_ftn(Q, R, csch),
		casted_unary_operation_ftn(R, R, csch)
	}},
	{"sech", {
		casted_unary_operation_ftn(Z, R, sech),
		casted_unary_operation_ftn(Q, R, sech),
		casted_unary_operation_ftn(R, R, sech)
	}},
	{"coth", {
		casted_unary_operation_ftn(Z, R, coth),
		casted_unary_operation_ftn(Q, R, coth),
		casted_unary_operation_ftn(R, R, coth)
	}},
	{"ln", {
		casted_unary_operation_ftn(Z, R, std::log),
		casted_unary_operation_ftn(Q, R, std::log),
		casted_unary_operation_ftn(R, R, std::log)
	}},
	{"lg", {
		casted_unary_operation_ftn(Z, R, lg),
		casted_unary_operation_ftn(Q, R, lg),
		casted_unary_operation_ftn(R, R, lg)
	}},
	{"log", {
		casted_unary_operation_ftn(Z, R, std::log10),
		casted_unary_operation_ftn(Q, R, std::log10),
		casted_unary_operation_ftn(R, R, std::log10),
		het_binary_operation_ftn(Z, Q, R, log),
		het_binary_operation_ftn(Z, R, R, log),
		het_binary_operation_ftn(Q, R, R, log)
	}},
	{{"int"}, {
		unary_operation_ftn(Z, Z, int),
		unary_operation_ftn(Q, Z, int),
		unary_operation_ftn(R, Z, int)
	}},
	{{"float"}, {
		unary_operation_ftn(Z, R, (long double)),
		unary_operation_ftn(Q, R, (long double)),
		unary_operation_ftn(R, R, (long double))
	}},
	{{"!"}, {
		unary_operation_ftn(Z, Z, utility::integral_factorial)
	}},
	{{"^T"}, {
		unary_operation_blank(VecZ, MatZ, a->get().transpose()),
		unary_operation_blank(VecQ, MatQ, a->get().transpose()),
		unary_operation_blank(VecR, MatR, a->get().transpose()),
		unary_operation_blank(MatZ, MatZ, a->get().transpose()),
		unary_operation_blank(MatQ, MatQ, a->get().transpose()),
		unary_operation_blank(MatR, MatR, a->get().transpose())
	}},
	{{"."}, {
		binary_operation_ftn(VecZ, VecZ, Z, inner),
		binary_operation_ftn(VecQ, VecQ, Q, inner),
		binary_operation_ftn(VecR, VecR, R, inner),
		binary_operation_ftn(VecCmpZ, VecCmpZ, CmpZ, inner),
		binary_operation_ftn(VecCmpQ, VecCmpQ, CmpQ, inner),
		binary_operation_ftn(VecCmpR, VecCmpR, CmpR, inner),
		het_binary_operation_ftn(VecZ, VecQ, Q, inner),
		het_binary_operation_ftn(VecZ, VecR, R, inner)
	}}
};

}

}