#ifndef OPERATION_H_
#define OPERATION_H_

// C++ Standard Libraries
#include <functional>
#include <string>
#include <vector>

// Engine Headers
#include <operand.hpp>

namespace zhetapi {

class operation : public Token {
public:
	// Aliases
	using mapper = std::function <Token *(const std::vector <Token *> &)>;
protected:
	std::string	__input		= "";
	std::string	__output	= "";
	std::size_t	__ops		= 0;
	mapper		__opn;
public:
	operation();
	operation(const std::string &, const std::string &,
			std::size_t,  mapper);

	Token *compute(const std::vector <Token *> &) const;

	type caller() const override;
	std::string str() const override;
	Token *copy() const override;
	bool operator==(Token *) const override;

	class bad_input_size {};
};

}

#endif
