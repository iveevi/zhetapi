#ifndef ALGORITHM_H_
#define ALGORITHM_H_

// C++ headers
#include <string>
#include <vector>

// Engine headers
#include <token.hpp>

#include <core/node_manager.hpp>

namespace zhetapi {

class algorithm : public Token {
	std::string					__ident;

	std::vector <std::string>			__args;

	// Compile when called on
	std::vector <std::string>			__statements;
public:
	algorithm(std::string, const std::vector <std::string> &,
			const std::vector <std::string> &);

	type caller() const override;
	Token *copy() const override;
	std::string str() const override;

	virtual bool operator==(Token *) const override;
};

}

#endif