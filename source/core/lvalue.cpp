#include <core/lvalue.hpp>

#include <engine.hpp>

namespace zhetapi {

lvalue::lvalue() {}

lvalue::lvalue(const std::string &symbol, Engine *context)
		: _symbol(symbol), _context(context) {}

void lvalue::assign(Token *tptr)
{
	Token *self = _context->get(_symbol);

	// Assumes either function or variable (variable for now)
	if (self->caller() == var)
		_context->put(Variable(tptr, _symbol));
}

Token::type lvalue::caller() const
{
	return Token::token_lvalue;
}

Token *lvalue::copy() const
{
	return new lvalue(_symbol, _context);
}

std::string lvalue::str() const
{
	return "lvalue-\"" + _symbol + "\"";
}

bool lvalue::operator==(Token *tptr) const
{
	lvalue *lv = dynamic_cast <lvalue *> (tptr);

	if (lv == nullptr)
		return false;

	return (lv->_symbol == _symbol)
		&& (lv->_context == _context);
}

}
