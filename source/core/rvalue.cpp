#include <core/rvalue.hpp>

#include <engine.hpp>

namespace zhetapi {

rvalue::rvalue() {}

rvalue::rvalue(const std::string &symbol, Engine *context)
		: _symbol(symbol), _context(context) {}

Token *rvalue::get() const
{
	// TODO: Maybe warn for null?
	Token *tptr = _context->get(_symbol);

	// Only constant for now
	if (tptr->caller() == Token::var)
		return (dynamic_cast <Variable *> (tptr))->get();

	return nullptr;
}

Token::type rvalue::caller() const
{
	return Token::token_rvalue;
}

Token *rvalue::copy() const
{
	return new rvalue(_symbol, _context);
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

	return (rv->_symbol == _symbol)
		&& (rv->_context == _context);
}

}
