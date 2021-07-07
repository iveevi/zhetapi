#include "../../engine/core/operation_base.hpp"

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

// Operations
std::unordered_map <std::string, Overload> operations {
	{"+", {
		casted_binary_operation_set(+),
		casted_vector_binary_operation_set(+)
	}},
	{"-", {casted_binary_operation_set(-)}},
	{"*", {casted_binary_operation_set(*)}},
	{"/", {casted_binary_operation_set(/)}},
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
	}}
};

}

}