#ifndef VARIALBE_H_
#define VARIALBE_H_

// C/C++ headers
#include <string>
#include <iostream>
#include <sstream>
#include <memory>

// Engine headers
#include <token.hpp>

namespace zhetapi {

class Variable : public Token {
	std::string		__symbol;
	std::shared_ptr <Token>	__tptr;
public:
	// Constructor
	Variable(Token * = nullptr, const ::std::string & = "");

	Variable(const Variable &);

	// Copy
	Variable &operator=(const Variable &);

	// Reference
	std::shared_ptr <Token> &get();
	const std::shared_ptr <Token> &get() const;

	const std::string &symbol() const;
	
	// Virtual functions
	Token::type caller() const override;
	std::string str() const override;
	Token *copy() const override;
	bool operator==(Token *) const override;

	// Exceptions
	class illegal_type {};
};
	
}

#endif
