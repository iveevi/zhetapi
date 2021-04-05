#include <core/rvalue.hpp>

#include <engine.hpp>

namespace zhetapi {

rvalue::rvalue() {}

rvalue::rvalue(const std::string &symbol)
		: _symbol(symbol) {}

const std::string &rvalue::symbol() const
{
	return _symbol;
}

Token *rvalue::get(Engine *context) const
{
	// TODO: Maybe warn for null?
	return context->get(_symbol);
}

Token::type rvalue::caller() const
{
	return Token::token_rvalue;
}

Token *rvalue::copy() const
{
	return new rvalue(_symbol);
}

std::string rvalue::str() const
{
	return "rvalue-\"" + _symbol + "\"";
}

bool rvalue::operator==(Token *tptr) const
{
	rvalue *rv = dynamic_cast <rvalue *> (tptr);

	if (rv == nullptr)
		return false;

	return rv->_symbol == _symbol;
}

}
