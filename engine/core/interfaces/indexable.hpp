#ifndef INDEXABLE_H_
#define INDEXABLE_H_

// Engine headers
#include "../../token.hpp"

namespace zhetapi {

// Indexable interface: indexing operator

/**
 * @brief An interface for any primitive type that can be indexed with the
 * bracket operator, [].
 */
class Indexable : public virtual Token {
public:
	/**
	 * @brief The indexing function that must be overriden by any primitive
	 * type that is to be indexable.
	 * 
	 * @param tptr the "index" to be taken.
	 * @param cref determines whether to return a reference to the indexed
	 * value.
	 * 
	 * @return the value at the specified "index".
	 */
	virtual Token *index(Token *tptr, bool ref = false) = 0;
};

}

#endif
