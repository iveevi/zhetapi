#include <token.hpp>

namespace zhetapi {

Token::~Token() {}

Token *Token::attr(const std::string &id, const std::vector <Token *> &args)
{
	if (args.size() == 0) {
		if (_attributes.find(id) == _attributes.end())
			throw unknown_attribute(id);
		
		return _attributes[id];
	} else {
		if (_methods.find(id) == _methods.end())
			throw unknown_attribute(id);
		
		return (_methods[id])(args);
	}
}

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
