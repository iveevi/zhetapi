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
	std::string	_symbol	= "";		// The actual symbol
public:
	rvalue();
	rvalue(const std::string &);

	// Properties
	const std::string &symbol() const;

	// Different from lvalue (the only difference)
	Token *get(Engine *) const;

	type caller() const override;
	Token *copy() const override;
	std::string str() const override;

	virtual bool operator==(Token *) const override;
};

}

#endif
