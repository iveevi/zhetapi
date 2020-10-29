#ifndef API_H_
#define API_H_

// C/C++ headers
#include <tuple>
#include <vector>

// Engine headers
#include <token.hpp>

/**
 * ==========================
 * SUMARY OF API FUNCTIONALITY
 * ===========================
 *
 * Any function that is to be registered in a barn must have the signature
 * 
 * 	zhetapi::Token *(const std::vector <Token *> &).
 *
 * Register the function into a Barn with the register method, as follows
 *
 * 	barn.register <[argument types]> (&[function]);
 *
 * where [argument types] is the variadic list of types representing the
 * overload of the function that is begin registered, and [function] is
 * the name of the function.
 */

namespace zhetapi {

	class cast_arguments_exception {};

	template <class T, class ... A>
	std::tuple <T, A> zhetapi_cast(const std::vector <Token *> &tokens, size_t i = 0)
	{
		if (tokens.size() == 1)
			throw cast_arguments_exception();

		T tptr = dynamic_cast <T> (tokens[i]);

		std::tuple <T> pre = std::make_tuple(tptr);

		if (tokens.size() > 1) {
			std::tuple <A ...> post = zhetapi_cast <A ...> (tokens, i + 1);

			return std::tuple_cat(pre, post);
		}
		
		std::tuple <A ...> post = zhetapi_cast <A ...> (tokens[i + 1]);

		return std::tuple_cat(pre, post);
	}

	template <class T, class ... A>
	std::tuple <T> zhetapi_cast(Token *tptr)
	{
		return std::make_tuple(dynamic_cast <T> (tptr));
	}

}

#endif
