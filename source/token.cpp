#include <token.hpp>

namespace zhetapi {

	Token::operator type() const
	{
		return caller();
	}

	bool Token::operator!=(Token *tptr) const
	{
		return !(*this == tptr); 
	}

}