#ifndef GENERATOR_H_
#define GENERATOR_H_

// Engine headers
#include "set.hpp"
#include "iterator.hpp"

namespace zhetapi {

/**
 * @brief Generator interface. Allows the inherited primitive types to be
 * iterated through using for loops and "in" syntax.
 * 
 * ie. allows for:
 * 
 * \code{.cpp}
 * for ([var] in [Generator])
 * 	[code]
 * \endcode
 */
class Generator : public Set {
public:
	/**
	 * @return the starting iterator.
	 */
	virtual Iterator *begin() const = 0;

	/**
	 * @brief Get the next iterator.
	 * 
	 * @param itr the current iterator.
	 * 
	 * @return the iterator ahead of the one passed, or nullptr if it was
	 * the end.
	 */
	virtual Iterator *next(Iterator *itr) = 0;
};

}

#endif