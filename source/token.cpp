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

std::ostream &operator<<(std::ostream &os, const std::vector <Token *> &toks)
{
	os << "{";

	size_t n = toks.size();
	for (size_t i = 0; i < n; i++) {
		os << toks[i]->str();

		if (i < n - 1)
			os << ", ";
	}

	os << "}";

	return os;
}


}
