#include "../../engine/engine.hpp"
#include "../../engine/core/lvalue.hpp"

namespace zhetapi {

lvalue::lvalue(const std::string &symbol)
		: _symbol(symbol) {}

const std::string &lvalue::symbol() const
{
	return _symbol;
}

void lvalue::assign(Token *tptr, Engine *context)
{
	if (context)
		context->put(_symbol, tptr);
}

Token::type lvalue::caller() const
{
	return Token::token_lvalue;
}

Token *lvalue::copy() const
{
	return new lvalue(_symbol);
}

std::string lvalue::dbg_str() const
{
	return "lvalue-\"" + _symbol + "\"";
}

bool lvalue::operator==(Token *tptr) const
{
	lvalue *lv = dynamic_cast <lvalue *> (tptr);

	if (lv == nullptr)
		return false;

	return lv->_symbol == _symbol;
}

}
