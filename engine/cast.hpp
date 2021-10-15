#ifndef CAST_H_
#define CAST_H_

// Engine headers
#include "token.hpp"

/**
 * @file cast.hpp
 * @brief Contains functions to help casting pointers to Tokens using
 * `dynamic_cast`. Helpful for dealing with different overloads in Registrables.
 * For example, suppose we have the following Registrable that is supposed to
 * take up to three integers.
 *
 * \code{.cpp}
 *
 * ZHETAPI_REGISTRABLE(my_registrable)
 * {
 * 	OpZ o1, o2, o3;
 *
 * 	// Performing overload switching with zhetapi_cast (should be used for very
 * 	// specific or seemingly random overloads)
 * 	if (zhetapi_cast(inputs, o1, o2, o3)) {
 * 		// do the function for three integers (o1, o2, o3)
 * 	} else if (zhetapi_cast(inputs, o1, o2)) {
 * 		// do the function for two integers (o1, o2)
 * 	} else if (zhetapi_cast(inputs, o1)) {
 * 		// do the function for two integers (o1)
 * 	} else {
 * 		// Terminating branch...
 * 	}
 *
 * 	// ...or use zhetapi_cc_cast (should be used for sequences of partial
 * 	// overloads)
 * 	switch (zhetapi_cc_cast(inputs, o1, o2, o3)) {
 * 	case 3:
 * 		// do the function for three integers (o1, o2, o3)
 * 	case 2:
 * 		// do the function for two integers (o1, o2)
 * 	case 1:
 * 		// do the function for one integer (o1)
 * 	default:
 * 		break;
 * 	}
 *
 * 	// As the terminating action either
 * 	return nullptr;
 *
 * 	// ...or throw
 * 	throw my_exception();
 * }
 *
 * \endcode
 */

namespace zhetapi {

// TODO: rename cc cast to partial cast
// and add another cast method that goes down the list until fully casted
template <class T>
bool zhetapi_cast_process(
		const std::vector <Token *> &tokens,
		size_t i,
		T &tptr)
{
	if (i >= tokens.size())
		return false;

	tptr = dynamic_cast <T> (tokens[i]);

	if (!tptr)
		return false;

	return true;
}

#ifndef SKIP_DOXYGEN	// Breathe cannot parse variadics

template <class T, class ... A>
bool zhetapi_cast_process(
		const std::vector <Token *> &tokens,
		size_t i,
		T &tptr,
		A &... args)
{
	if (i >= tokens.size())
		return false;

	tptr = dynamic_cast <T> (tokens[i]);

	if (!tptr)
		return false;

	return zhetapi_cast_process(tokens, i + 1, args ...);
}

template <class ... A>
bool zhetapi_cast(const std::vector <Token *> &tokens, A &... args)
{
	return zhetapi_cast_process(tokens, 0, args ...);
}

#endif

// Counting alternatives
template <class T>
void zhetapi_cast_cc_process(
		const std::vector <Token *> &tokens,
		size_t &i,
		T &tptr)
{
	if (i >= tokens.size())
		return;

	tptr = dynamic_cast <T> (tokens[i]);

	if (!tptr)
		return;

	i++;
}

#ifndef SKIP_DOXYGEN	// Breathe cannot parse variadics

template <class T, class ... A>
void zhetapi_cast_cc_process(
		const std::vector <Token *> &tokens,
		size_t &i,
		T &tptr,
		A &... args)
{
	if (i >= tokens.size())
		return;

	tptr = dynamic_cast <T> (tokens[i]);

	if (!tptr)
		return;

	zhetapi_cast_cc_process(tokens, ++i, args ...);
}

template <class ... A>
size_t zhetapi_cast_cc(const std::vector <Token *> &tokens, A &... args)
{
	size_t success = 0;
	zhetapi_cast_cc_process(tokens, success, args ...);
	return success;
}

#endif

}

#endif
