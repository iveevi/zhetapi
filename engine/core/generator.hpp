#ifndef GENERATOR_H_
#define GENERATOR_H_

// Engine headers
#include "set.hpp"
#include "iterator.hpp"

namespace zhetapi {

// Generator interface: for-"in"
class Generator : public Set {
public:
	// Return iterators
	virtual Iterator *begin() const = 0;
	virtual Iterator *next(Iterator *) = 0;
};

}

#endif