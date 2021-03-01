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
	std::string	__symbol	= "";
	Token *		__tptr		= nullptr;
public:
	// Memory and initialization
	Variable();
	Variable(const Variable &);
	Variable(Token *, const std::string &);

	Variable &operator=(const Variable &);

	~Variable();

	// Properties
	Token *get();
	const std::string &symbol() const;
	
	// Virtual functions
	Token::type caller() const override;
	std::string str() const override;
	Token *copy() const override;
	bool operator==(Token *) const override;
};
	
}

#endif
