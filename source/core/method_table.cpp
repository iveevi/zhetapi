#include "../../engine/core/method_table.hpp"

namespace zhetapi {

MethodTable::MethodTable(Table table) : _mtable(table) {}

Token *MethodTable::get(
		const std::string &id,
		Token *tptr,
		Engine *ctx,
		const Targs &args) const
{
	if (_mtable.find(id) != _mtable.end())
		return _mtable.at(id).method(tptr, ctx, args);

	throw Token::unknown_attribute(id);
}

const std::string &MethodTable::docs(const std::string &id) const
{
	if (_mtable.find(id) != _mtable.end())
		return _mtable.at(id).docs;

	// TODO: should we differentiate from method?
	throw Token::unknown_attribute(id);
}

void MethodTable::list(std::ostream &os) const
{
	os << "Methods:" << std::endl;
	for (const auto &m : _mtable)
		os << "\t" << m.first << std::endl;
}

}
