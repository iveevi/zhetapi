#include <token.hpp>

#include <core/types.hpp>

namespace zhetapi {

Token::Token() {}

Token::Token(const std::vector <std::pair <std::string, method>> &attrs)
{
	for (auto attr_pr : attrs)
		this->_attributes[attr_pr.first] = attr_pr.second;
}

Token::~Token() {}

Token *Token::attr(const std::string &id, const std::vector <Token *> &args)
{
	if (_attributes.find(id) == _attributes.end())
		throw unknown_attribute(typeid(*this), id);
		
	return _attributes[id](this, args);
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
		os << toks[i]->dbg_str();

		if (i < n - 1)
			os << ", ";
	}

	os << "}";

	return os;
}

// Unknown attribute exception
 std::string Token::unknown_attribute::what() const
 {
	// TODO: use the actual name instead of mangled
	return "<" + type_name(_ti) + ">"
		+ " has no attribute \""
		+ _msg + "\"";
}

size_t Token::id() const
{
	return 0;
}

Token::type Token::caller() const
{
	return undefined;
}

std::string Token::dbg_str() const
{
	return "[?]";
}

}
