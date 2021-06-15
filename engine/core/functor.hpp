#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C/C++ headers
#include <exception>

// Engine headers
#include "../token.hpp"
#include "common.hpp"

namespace zhetapi {

class Engine;

// Functor interface
class Functor : public virtual Token {
public:
	virtual Token *evaluate(Engine *, const Targs &) = 0;

	// Exception
	class insufficient_args : public std::runtime_error {
	public:
		insufficient_args(size_t got, size_t need)
			: std::runtime_error("Needed " + std::to_string(need)
				+ " arguments, but received " + std::to_string(got)
				+ " instead") {}
	};
};

}

#endif
