#ifndef ASSIGNABLE_H_
#define ASSIGNABLE_H_

// Engine headers
#include "../../token.hpp"

namespace zhetapi {

class Assignable : public virtual Token {
public:
	virtual void assign(Token *) = 0;
};

}

#endif
