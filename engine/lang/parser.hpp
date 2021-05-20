#ifndef PARSER_H_
#define PARSER_H_

// C/C++ headers
#include <iostream>

// Engine headers
#include "feeder.hpp"
#include "../engine.hpp"

namespace zhetapi {

// Parsing state
struct State {
	std::string	cached;

	// Nesting states
	size_t		parenthesis	= 0;
	size_t		braces		= 0;
	size_t		brackets	= 0;

	bool		branch		= false;
	bool		bdone		= false;

	bool		alg		= false;

	State(bool = false);

	void check_nest(char);
	bool is_nested();
};

void run(const std::string &, Engine *);

void check_keyword(std::string &, Feeder *, Engine *, State *);

// Make part of public API so others and I can use
int parse_global(const std::string &, Engine *);
int parse_global(Feeder *, Engine *);

}

#endif
