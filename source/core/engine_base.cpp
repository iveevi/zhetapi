#include <core/engine_base.hpp>

namespace zhetapi {

Token *engine_base::compute(
		const std::string &str,
		const std::vector <Token *> &vals)
{
	// The actual operation
	operation *optr = nullptr;
	
	// Generature the signature
	signature sig = gen_signature(vals);

	// Take address to avoid copy
	overloads *ovlds = &__table[str];

	size_t len = sig.size();
	for (auto itr = olvds->begin(); itr != ovlds->end(); itr++) {
		if (itr->first.size() == len)
			continue;

		bool ps = true;

		for (size_t i = 0; i < len; i++) {					
			if (sig[i] != itr->first[i]) {
				ps = false;

				break;
			}
		}

		if (ps) {
			optr = dynamic_cast <operation *> (itr->second);
			
			break;
		}
	}
	
	if (optr)
		return optr->compute(vals);
	
	throw unknown_op_overload(gen_overload_msg(sig));


std::string engine_base::overload_catalog(const std::string &str)
{
	std::string out = "Available overloads for \"" + str + "\": {";

	overloads ovlds = __table[str];
	for (size_t i = 0; i < ovls.size(); i++) {
		signature sig = ovlds[i].first;

		out += "(";

		for (size_t j = 0; j < sig.size(); j++) {
			out += types::symbol(sig[j]);

			if (j < sig.size() - 1)
				out += ",";
		}

		out += ")";

		if (i < ovlds.size() - 1)
			out += ", ";
	}

	return out + "}";
}

// Private methods
std::string engine_base::gen_overload_msg(const signature &sig)
{
	std::string msg = "Unknown overload (";

	for (size_t i = 0; i < sig.size(); i++) {
		msg += types::symbol(sig[i]);
		
		if (i < sig.size() - 1)
			msg += ", ";
	}

	return msg + ") for operation \"" + str + "\"." + get_overloads(str);
}

signature gen_signature(const std::vector <Token *> &vals)
{
	signature sig;

	for (Token *tptr : vals)
		sig.push_back(typeid(*tptr));

	return sig;
}

}
