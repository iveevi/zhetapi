#ifndef LVALUE_H_
#define LVALUE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <token.hpp>

namespace zhetapi {

// Forward declaration
class Engine;

class lvalue : public Token {
protected:
	std::string	_symbol	= "";		// The actual symbol
public:
	lvalue();
	lvalue(const std::string &);

	// Properties
	const std::string &symbol() const;

	void assign(Token *, Engine *);

	type caller() const override;
	Token *copy() const override;
	std::string dbg_str() const override;

	virtual bool operator==(Token *) const override;
};

}

#endif
