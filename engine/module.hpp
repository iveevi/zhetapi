#ifndef MODULE_H_
#define MODULE_H_

// C/C++ headers
#include <regex>
#include <unordered_map>

// Engine headers
#include "token.hpp"
#include "core/common.hpp"

namespace zhetapi {

using NamedToken = std::pair <std::string, Token *>;

/**
 * @brief Represents a collection of variables (algorithms, registrables,
 * functions, operands) from a separate library or file.
 */
class Module : public Token {
	std::string					_name;

	// Add documentation for attributes as well
	std::unordered_map <std::string, Token *>	_attributes;
public:
	Module(const std::string &);
	Module(const std::string &, const std::vector <NamedToken> &);

	// Overriding attr
	Token *attr(const std::string &, Engine *, const Targs &, size_t) override;

	void list_attributes(std::ostream &) const override;

	// Methods
	void add(const NamedToken &);
	void add(const char *, Token *);
	void add(const std::string &, Token *);

	void from_add(Engine *, const Args &);

	// Virtual functions
	virtual type caller() const override;
	virtual std::string dbg_str() const override;
	virtual Token *copy() const override;
	virtual bool operator==(Token *) const override;

	/**
	 * @brief Exporter type alias. The "exporter" of each compiled library is a
	 * function that is executed upon loading the library. Any symbols that the
	 * library designer wishes to make available through the library's interface
	 * are made public in this function.
	 */
	using Exporter = void (*)(Module *);
};

/**
 * @brief Defines the exporter function of a library. There should exactly be
 * one such function for each compiled library.
 */
#define ZHETAPI_LIBRARY()				\
	extern "C" void zhetapi_export_symbols(zhetapi::Module *module)

/**
 * @brief Defines a function with signature Token *(const std::vector <Token *> &).
 * 
 * @param fident the name of the function.
 */
#define ZHETAPI_REGISTER(fident)			\
	zhetapi::Token *fident(const std::vector <zhetapi::Token *> &inputs)

// TODO: add an overload for registrables, and others

/**
 * @brief Adds a Registrable to the module (inside the zhetapi_export_function)
 * 
 * @param symbol the name of the function to add.
 */
#define ZHETAPI_EXPORT(symbol)				\
	module->add(#symbol, new zhetapi::Registrable(#symbol, &symbol));

/**
 * @brief Adds a Registrable to the module (inside the zhetapi_export_function)
 * 
 * @param symbol the name of that the function should have in the module (the
 * binded name).
 * @param ftr the name of the function to bind.
 */
#define ZHETAPI_EXPORT_SYMBOL(symbol, ftr)		\
	module->add(#symbol, new zhetapi::Registrable(#symbol, &ftr));

/**
 * @brief Adds an operand type variable to the module.
 * 
 * @param symbol the name that the value should have (the binded name).
 * @param type the type of the value.
 * @param op the actual value of the variable.
 */
#define ZHETAPI_EXPORT_CONSTANT(symbol, type, op)	\
	module->add(#symbol, new zhetapi::Operand <type> (op));

}

#endif
