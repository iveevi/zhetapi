#ifndef ITERATOR_H_
#define ITERATOR_H_

// Engine headers
#include "../token.hpp"

namespace zhetapi {

/**
 * @brief Iterator interface for Generator types.
 */
class Iterator : public Token {
public:
	/**
	 * @return the value of the iterator.
	 */
	virtual Token *value() const = 0;
};

}

#endif