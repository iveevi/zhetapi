#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// Engine headers
#include "../token.hpp"
#include "common.hpp"

namespace zhetapi {

class Engine;

// Functor interface
class Functor : public virtual Token {
public:
	virtual Token *evaluate(Engine *, const Targs &);
}

}

#endif