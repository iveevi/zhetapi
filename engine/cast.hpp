#ifndef CAST_H_
#define CAST_H_

// Engine headers
#include "token.hpp"

namespace zhetapi {

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

}

#endif
