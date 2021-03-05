#include <core/rvalue.hpp>

#include <barn.hpp>

namespace zhetapi {

rvalue::rvalue() {}

rvalue::rvalue(const std::string &symbol, Barn *context)
		: __symbol(symbol), __context(context) {}

Token *rvalue::get() const
{
	// TODO: Maybe warn for null?
	Token *tptr = __context->get(__symbol);

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
	return new rvalue(__symbol, __context);
}

std::string rvalue::str() const
{
	return "rvalue-\"" + __symbol + "\"";
}

bool rvalue::operator==(Token *tptr) const
{
	rvalue *rv = dynamic_cast <rvalue *> (tptr);

	if (rv == nullptr)
		return false;

	return (rv->__symbol == __symbol)
		&& (rv->__context == __context);
}

}
