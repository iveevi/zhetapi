#ifndef INDEXABLE_H_
#define INDEXABLE_H_

// Engine headers
#include "../token.hpp"

namespace zhetapi {

// Indexable interface: indexing operator
class Indexable : public virtual Token {
public:
	virtual Token *index(Token *) = 0;
};

}

#endif