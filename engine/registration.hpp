#ifndef REGISTRATION_H_
#define REGISTRATION_H_

// C/C++ headers
#include <tuple>
#include <vector>
#include <functional>

// Engine headers
#include "core/functor.hpp"

// TODO: separate casting from registration

/**
 * This file contains functions that aid the creation of Engine-registrable
 * functions.
 *
 * ======================
 * SUMARY OF REGISTRATION
 * ======================
 *
 * Any function that is to be registered in a engine must have the signature
 *
 * 	zhetapi::Token *(const std::vector <Token *> &).
 *
 * Register the function into a Engine with the register method, as follows
 *
 * 	engine.register <[argument types]> (&[function]);
 *
 * where [argument types] is the variadic list of types representing the
 * overload of the function that is begin registered, and [function] is
 * the name of the function.
 */

namespace zhetapi {

class Registrable : public Functor {
public:
	using Evaluator = std::function <Token *(const std::vector <Token *> &)>;
private:
	Evaluator	_ftn;

	std::string	_ident;
public:
	Registrable();
	Registrable(const Registrable &);
	Registrable(const std::string &, Evaluator);

	// TODO: get rid of this
	Token *operator()(const std::vector <Token *> &) const;

	Token *evaluate(Engine *, const std::vector <Token *> &) override;

	std::string dbg_str() const override;
	type caller() const override;
	Token *copy() const override;
	bool operator==(Token *) const override;
};

/**
 * ==============================
 * ZHETAPI CAST NULLPTR EXCEPTION
 * ==============================
 *
 * This exception is thrown when the casting process (into a Zhetapi
 * Token type) fails. Technically speaking, this is not an exception,
 * but rather the feault of the user. Regardless, this exception is
 * thrown in such a situation, and gives the user a way to ensure that
 * the casting has failed, and act accordingly.
 */

class zhetapi_cast_nullptr {};

/**
 * ===============================
 * ZHETAPI CAST OVERFLOW EXCEPTION
 * ===============================
 *
 * This exception is thrown when the number of pointers that are to be
 * casted exceeds the number of Tokens in the passed vector. This is
 * thrown with the purpose of alterting the user of this situation.
 */

class zhetapi_cast_overflow {};

/**
 * ============
 * ZHETAPI CAST
 * ============
 *
 * Casts the Zhetapi Tokens into the passed POINTERS:
 *
 * zhetapi_cast(tokens, [ptr_1], [ptr_2], ... , [ptr_n])
 *
 * The function casts tokens[0] to a pointer of type decltype(ptr_1),
 * tokens[1] to a pointer of type decltype(ptr_2), and so on until
 * [ptr_n]. A zhetapi_cast_nullptr exception is thrown if a cast fails
 * (results in a null result), and a zhetapi_cast_overflow exception is
 * thrown if the number of pointers to cast to is larger than the size
 * of the vector passed in. Both these exceptions are thrown so that
 * users can handle such cases with ease (rather than checking each
 * casted pointer).
 */

template <class T>
void zhetapi_cast_process(
		const std::vector <Token *> &tokens,
		size_t i,
		T &tptr)
{
	if (i >= tokens.size())
		throw zhetapi_cast_overflow();

	tptr = dynamic_cast <T> (tokens[i]);

	if (!tptr)
		throw zhetapi_cast_nullptr();
}

template <class T, class ... A>
void zhetapi_cast_process(
		const std::vector <Token *> &tokens,
		size_t i,
		T &tptr,
		A &... args)
{
	if (i >= tokens.size())
		throw zhetapi_cast_overflow();

	tptr = dynamic_cast <T> (tokens[i]);

	if (!tptr)
		throw zhetapi_cast_nullptr();

	zhetapi_cast_process(tokens, i + 1, args ...);
}

template <class ... A>
void zhetapi_cast(const std::vector <Token *> &tokens, A &... args)
{
	zhetapi_cast_process(tokens, 0, args ...);
}

}

#endif
