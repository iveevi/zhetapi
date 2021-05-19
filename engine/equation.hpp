#ifndef EQUATION_H_
#define EQUATION_H_

// C/C++ headers
#include <exception>
#include <vector>

// Engine headers
#include "engine.hpp"

namespace zhetapi {

using Solutions = std::vector <std::vector <Token *>>;

class Equation {
	std::vector <node_manager>	_expressions	= {};
	std::vector <std::string>	_args		= {};
	Engine *			_engine		= nullptr;
public:
	Equation(const std::vector <std::string> &);

	// Properties
	size_t args() const;

	// Methods
	Solutions solve() const;

	std::string representation() const;

	// Exceptions
	class bad_input_size : std::runtime_error {
	public:
		bad_input_size()
			: std::runtime_error("Bad input size for equation") {}
	};
};

std::ostream &operator<<(std::ostream &, const Equation &);

}

#endif
