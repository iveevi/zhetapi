#ifndef GENERATOR_H_
#define GENERATOR_H_

// Engine headers
#include "set.hpp"

namespace zhetapi {

// Generator interface: for-"in"
class Generator : public Set {
public:
	virtual Token *next();
	virtual void reset();
};

}

#endif