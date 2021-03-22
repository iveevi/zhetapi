#include <core/wildcard.hpp>

namespace zhetapi {

wildcard::wildcard(const std::string &str, predicate pred)
		: __symbol(str), __pred(pred) {}

Token::type wildcard::caller() const
{
	return Token::token_wildcard;
}

Token *wildcard::copy() const
{
	return new wildcard(__symbol, __pred);
}

std::string wildcard::str() const
{
	return "w-\"" + __symbol + "\"";
}

bool wildcard::operator==(Token *tptr) const
{
	wildcard *wld = dynamic_cast <wildcard *> (tptr);

	if (wld == nullptr)
		return false;

	return true;
}

}