#ifndef MODULE_H_
#define MODULE_H_

// Engine headers
#include "token.hpp"

namespace zhetapi {

using NamedToken = std::pair <std::string, Token *>;

class Module : public Token {
	std::string	_name;
public:
	Module(const std::string &);
	Module(const std::string &, const std::vector <NamedToken> &);

	// Methods
	void add(const NamedToken &);
	void add(const char *, Token *);
	void add(const std::string &, Token *);

	// Virtual functions
	virtual type caller() const;
	virtual uint8_t id() const;
	virtual std::string dbg_str() const;
	virtual Token *copy() const;
	virtual bool operator==(Token *) const;
};

// Aliases
using Exporter = void (*)(Module *);

// Library creation macros
#define ZHETAPI_LIBRARY()				\
	extern "C" void zhetapi_export_symbols(zhetapi::Module *module)

#define ZHETAPI_REGISTER(fident)			\
	zhetapi::Token *fident(const std::vector <zhetapi::Token *> &inputs)

#define ZHETAPI_EXPORT(symbol)				\
	module->add(#symbol, new zhetapi::Registrable(#symbol, &symbol));

#define ZHETAPI_EXPORT_SYMBOL(symbol, ftr)		\
	module->add(#symbol, new zhetapi::Registrable(#symbol, &ftr));

#define ZHETAPI_EXPORT_CONSTANT(symbol, type, op)	\
	module->add(#symbol, new zhetapi::Operand <type> (op));


}

#endif
