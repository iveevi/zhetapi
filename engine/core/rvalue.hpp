#ifndef RVALUE_H_
#define RVALUE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <token.hpp>
#include <variable.hpp>

// Create as middle class inheriter
// ('reference', then branch into rvalue and lvalue)
// Then combine into a single header
namespace zhetapi {

// Forward declaration
class Engine;

class rvalue : public Token {
protected:
	std::string	__symbol	= "";		// The actual symbol
	Engine *	__context	= nullptr;	// Acts as the scope
public:
	rvalue();
	rvalue(const std::string &, Engine *);

	// Different from lvalue (the only difference)
	Token *get() const;

	type caller() const override;
	Token *copy() const override;
	std::string str() const override;

	virtual bool operator==(Token *) const override;
};

}

#endif
