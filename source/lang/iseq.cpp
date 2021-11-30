#include "../../engine/lang/iseq.hpp"

// Standard headers
#include <iomanip>

namespace zhetapi {

ISeq::ISeq(std::queue <Parser::TagPair> &postfix, const Args &args)
	: _nargs(args.size())
{
	// Allocate
	_args = new Variant[_nargs];

	// For now dont compress constant
	// TODO: need to fix the above - save space on constants...

	// Get index ftn
	auto arg_index = [&](const std::string &str) {
		for (int i = 0; i < args.size(); i++) {
			if (args[i] == str)
				return i;
		}

		return -1;
	};

	while (!postfix.empty()) {
		Parser::TagPair pr = postfix.front();
		postfix.pop();

		// std::cout << "pr = " << to_string(pr.data) << std::endl;

		if (pr.tag == PRIMITIVE) {
			Primitive prim = PrimitiveTag::cast(pr.data);
			size_t index = _consts.size();
			_consts.push_back(new Primitive(prim));
			_code.push_back(core::l_add);
			_code.push_back(index);
		} else if (pr.tag == IDENTIFIER) {
			// For now only variables
			std::string var = IdentifierTag::cast(pr.data);
			size_t index = arg_index(var);
			_code.push_back(core::l_get);
			_code.push_back(index);
		} else if (is_operation(pr.tag)) {
			_code.push_back(pr.tag);
		} else {
			// Throw
		}
	}

	// Zero the arguments
	for (size_t i = 0; i < _nargs; i++)
		_args[i] = new Primitive(0LL);
}

void ISeq::dump()
{
	// Dashes
	std::string dash1;
	std::string dash2;
	std::string dash3;
	std::string dash4;

	// Creating the dashes
	for (size_t i = 0; i < 7; i++) dash1 += "\u2500";
	for (size_t i = 0; i < 15; i++) dash2 += "\u2500";
	for (size_t i = 0; i < 15; i++) dash3 += "\u2500";
	for (size_t i = 0; i < 15; i++) dash4 += "\u2500";

	// Printing the header
	std::cout << "\u250C" << dash1 << "\u252C"
		<< dash2 << "\u252C" << dash3 << "\u252C"
		<< dash4 << "\u2510" << std::endl;

	std::cout << "\u2502 INDEX \u2502 " << "CODE\t\t\u2502 "
		<< "VREGS\t\t\u2502 " << "CREGS\t\t\u2502\n";

	std::cout << "\u251C" << dash1 << "\u253C"
		<< dash2 << "\u253C" << dash3 << "\u253C"
		<< dash4 << "\u2524" << std::endl;
	for (size_t i = 0, row = 0; i < _code.size(); i++, row++) {
		// Index
		std::cout << std::right << "\u2502" << std::setw(6) << row << " \u2502 ";

		// Instruction
		std::string str;
		if (_code[i] == 0)
			str = "GET   " + std::to_string((int) _code[++i]);
		else if (_code[i] == 1)
			str = "CONST " + std::to_string((int) _code[++i]);
		else
			str = strlex[LexTag(_code[i])];

		std::cout << std::left << std::setw(13) << str << " \u2502 ";

		// Arguments
		if (row < _nargs)
			std::cout << std::setw(13) << variant_str(_args[row]) << " \u2502 ";
		else
			std::cout << std::setw(13) << "" << " \u2502 ";

		// Constants
		if (row < _consts.size())
			std::cout << std::setw(13) << variant_str(_consts[row]) << " \u2502 ";
		else
			std::cout << std::setw(13) << "" << " \u2502 ";

		// Newline
		std::cout << '\n';
	}
	std::cout << "\u2514" << dash1 << "\u2534"
		<< dash2 << "\u2534" << dash3 << "\u2534"
		<< dash4 << "\u2518" << std::endl;
}

}
