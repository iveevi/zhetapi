#include <core/lvalue.hpp>

#include <engine.hpp>

namespace zhetapi {

lvalue::lvalue() {}

lvalue::lvalue(const std::string &symbol, Engine *context)
		: __symbol(symbol), __context(context) {}

void lvalue::assign(Token *tptr)
{
	Token *self = __context->get(__symbol);

	// Assumes either function or variable (variable for now)
	if (self->caller() == var)
		__context->put(Variable(tptr, __symbol));
}

Token::type lvalue::caller() const
{
	return Token::token_lvalue;
}

Token *lvalue::copy() const
{
	return new lvalue(__symbol, __context);
}

std::string lvalue::str() const
{
	return "lvalue-\"" + __symbol + "\"";
}

bool lvalue::operator==(Token *tptr) const
{
	lvalue *lv = dynamic_cast <lvalue *> (tptr);

	if (lv == nullptr)
		return false;

	return (lv->__symbol == __symbol)
		&& (lv->__context == __context);
}

}
