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
	Token *tptr = context->get(_symbol);

	// Get the right line number
	if (!tptr) {
		context->list();
		throw node_manager::undefined_symbol(_symbol);
	}

	return tptr;
}

Token::type rvalue::caller() const
{
	return Token::token_rvalue;
}

Token *rvalue::copy() const
{
	return new rvalue(_symbol);
}

std::string rvalue::dbg_str() const
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
