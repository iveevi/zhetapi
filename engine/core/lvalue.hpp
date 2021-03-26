#ifndef LVALUE_H_
#define LVALUE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <token.hpp>
#include <variable.hpp>

namespace zhetapi {

// Forward declaration
class Engine;

class lvalue : public Token {
protected:
	std::string	_symbol	= "";		// The actual symbol
	Engine *	_context	= nullptr;	// Acts as the scope
public:
	lvalue();
	lvalue(const std::string &, Engine *);

	void assign(Token *);

	type caller() const override;
	Token *copy() const override;
	std::string str() const override;

	virtual bool operator==(Token *) const override;
};

}

#endif
