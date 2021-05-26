#ifndef ITERATOR_H_
#define ITERATOR_H_

// Engine headers
#include "../token.hpp"

namespace zhetapi {

class Iterator : public Token {
public:
	virtual Token *value() const = 0;
};

}

#endif