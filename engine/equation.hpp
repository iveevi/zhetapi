#ifndef EQUATION_H_
#define EQUATION_H_

// C/C++ headers

// Engine headers
#include <engine.hpp>

namespace zhetapi {

using Solutions = std::vector <std::vector <Token *>>;

class Equation {
	std::vector <node_manager>	__expressions	= {};
	std::vector <std::string>	__args		= {};
	Engine *			__engine	= nullptr;
public:
	Equation(const std::vector <std::string> &);

	// Properties
	size_t args() const;

	// Methods
	Solutions solve() const;

	std::string representation() const;
};

std::ostream &operator<<(std::ostream &, const Equation &);

}

#endif
