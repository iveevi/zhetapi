#include "../engine/token.hpp"
#include "../engine/core/types.hpp"
#include "../engine/core/algorithm.hpp"

namespace zhetapi {

/**
 * @brief Default Token constructor that does nothing. For inheritance purposes.
 */
Token::Token() {}

Token::Token(const std::vector <std::pair <std::string, method>> &attrs)
{
	for (auto attr_pr : attrs)
		this->_methods[attr_pr.first] = attr_pr.second;
}

Token::~Token() {}

Token *Token::attr(Engine *context, const std::string &id, const std::vector <Token *> &args)
{
	// TODO: how to deal with functors?
	// Priorotize attributes
	if (_attributes.find(id) != _attributes.end()) {
		Token *tptr = _attributes[id];
		if (tptr->caller() == Token::alg) {
			// std::cout << "ALGORITHM attribute!" << std::endl;

			algorithm *alg = dynamic_cast <algorithm *> (tptr);
			return alg->evaluate(context, args);
		}

		return _attributes[id];
	}

	if (_methods.find(id) != _methods.end())
		return _methods[id](this, args);

	throw unknown_attribute(typeid(*this), id);

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
	os << "Methods:" << std::endl;
	for (const auto &m : _methods)
		os << "\t" << m.first << std::endl;

	os << "Attributes:" << std::endl;
	for (const auto &a : _attributes)
		os << "\t" << a.first << " = " << a.second->dbg_str() << std::endl;
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
const char *Token::unknown_attribute::what() const noexcept
{
	// TODO: use the actual name instead of mangled
	// TODO: use id() instead of typeid
	return ("<" + type_name(_ti) + ">"
		+ " has no attribute \""
		+ std::runtime_error::what() + "\"").c_str();
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
