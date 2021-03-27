#include <core/lvalue.hpp>

#include <engine.hpp>

namespace zhetapi {

lvalue::lvalue() {}

lvalue::lvalue(const std::string &symbol, Engine *context)
		: _symbol(symbol), _context(context) {}

const std::string &lvalue::symbol() const
{
	return _symbol;
}

// Is this lvalue simply a place holder?
bool lvalue::is_dummy() const
{
	return _context == nullptr;
}

void lvalue::assign(Token *tptr)
{
	Token *self = _context->get(_symbol);

	// Assumes either function or variable (variable for now)
	if (self->caller() == var && _context)
		_context->put(_symbol, tptr);
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
