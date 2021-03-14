#ifndef EQUATION_H_
#define EQUATION_H_

// C/C++ headers

// Engine headers
#include <engine.hpp>

namespace zhetapi {

class Equation {
	std::vector <node_manager>	__expressions	= {};
	Engine *			__engine	= nullptr;
public:
	Equation(const std::vector <std::string> &);
};

}

#endif