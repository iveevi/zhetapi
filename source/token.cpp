#include "../engine/token.hpp"
#include "../engine/core/method_table.hpp"
#include "../engine/core/types.hpp"
#include "../engine/core/algorithm.hpp"

namespace zhetapi {

/**
 * @brief Default Token constructor that does nothing. For inheritance purposes.
 */
Token::Token() {}

Token::Token(MethodTable *mtable) : _mtable(mtable) {}

Token::~Token() {}

Token *Token::attr(const std::string &id, Engine *ctx, const Targs &args)
{
	if (_mtable)
		return _mtable->get(id, this, ctx, args);

	throw unknown_attribute(id);

	return nullptr;
}

bool Token::operator!=(Token *tptr) const
{
	return !(*this == tptr);
}

bool tokcmp(Token *a, Token *b)
{
	return *a == b;
}

void Token::list_attributes(std::ostream &os) const
{
	if (_mtable)
		_mtable->list(os);
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

// Defaulting virtual functions
uint8_t Token::id() const
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

/**
 * @brief Writes data to an output stream. Not pure virtual because not all
 * Token classes need this.
 *
 * @param os the stream to be written to.
 */
void Token::write(std::ostream &os) const
{
	throw empty_io();
}

bool Token::operator==(Token *) const
{
	return false;
}

}
