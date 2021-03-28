#include <core/lvalue.hpp>

#include <engine.hpp>

namespace zhetapi {

lvalue::lvalue(const std::string &symbol)
		: _symbol(symbol) {}

const std::string &lvalue::symbol() const
{
	return _symbol;
}

void lvalue::assign(Token *tptr, Engine *context)
{
	using namespace std;
	cout << "LVALUE ASSIGNMENT:" << endl;
	context->list();
	if (context)
		context->put(_symbol, tptr);
	cout << "post:" << endl;
	context->list();
}

Token::type lvalue::caller() const
{
	return Token::token_lvalue;
}

Token *lvalue::copy() const
{
	return new lvalue(_symbol);
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

	return lv->_symbol == _symbol;
}

}
