#ifndef LVALUE_H_
#define LVALUE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <token.hpp>
#include <variable.hpp>

namespace zhetapi {

// Forward declaration
class Barn;

class lvalue : public Token {
protected:
	std::string	__symbol	= "";		// The actual symbol
	Barn *		__context	= nullptr;	// Acts as the scope
public:
	lvalue();
	lvalue(const std::string &, Barn *);

	void assign(Token *);

	type caller() const override;
	Token *copy() const override;
	std::string str() const override;

	virtual bool operator==(Token *) const override;
};

}

#endif
