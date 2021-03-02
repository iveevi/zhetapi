#include <token.hpp>

namespace zhetapi {

Token::~Token() {}

bool Token::operator!=(Token *tptr) const
{
	return !(*this == tptr); 
}

bool tokcmp(Token *a, Token *b)
{
	return *a == b;
}

}
