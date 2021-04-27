#include <core/wildcard.hpp>

namespace zhetapi {

wildcard::wildcard(const std::string &str, predicate pred)
		: _symbol(str), _pred(pred) {}

Token::type wildcard::caller() const
{
	return Token::token_wildcard;
}

Token *wildcard::copy() const
{
	return new wildcard(_symbol, _pred);
}

std::string wildcard::dbg_str() const
{
	return "w-\"" + _symbol + "\"";
}

bool wildcard::operator==(Token *tptr) const
{
	wildcard *wld = dynamic_cast <wildcard *> (tptr);

	if (wld == nullptr)
		return false;

	return true;
}

}