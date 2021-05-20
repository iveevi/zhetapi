#ifndef PARSER_H_
#define PARSER_H_

// C/C++ headers
#include <iostream>

// Engine headers
#include "feeder.hpp"
#include "../engine.hpp"

namespace zhetapi {

void run(const std::string &, Engine *);

// Make part of public API so others and I can use
int parse_global(const std::string &, Engine *);
int parse_global(Feeder *, Engine *);

}

#endif
