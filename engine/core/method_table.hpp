#ifndef METHOD_TABLE_H_
#define METHOD_TABLE_H_

// C/C++ headers
#include <string>
#include <unordered_map>
#include <vector>

// Engine headers
#include "../token.hpp"
#include "common.hpp"

namespace zhetapi {

// Forward declarations
class Engine;

// MethodTable class
class MethodTable {
public:
	using Method = Token *(*)(Token *, Engine *, const std::vector <Token *> &);

	struct MethodEntry {
		Method		method;
		std::string	docs;
	};

	using Table = std::unordered_map <std::string, MethodEntry>;
private:
	Table _mtable;
public:
	MethodTable();
	MethodTable(Table);

	Token *get(const std::string &, Token *, Engine *, const Targs &) const;
	const std::string &docs(const std::string &) const;
	void list(std::ostream &) const;
};

// Method creation macros
#define ZHP_TOKEN_METHOD(name)				\
	Token *name(Token *tptr, Engine *ctx, const Targs &args)

}

#endif
