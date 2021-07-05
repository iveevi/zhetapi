#ifndef ENGINE_BASE_H_
#define ENGINE_BASE_H_

// C/C++ headers
#include <stdexcept>
#include <unordered_map>
#include <typeinfo>
#include <typeindex>
#include <vector>

// Engine headers
#include "../token.hpp"
#include "../std/combinatorial.hpp"
#include "operation.hpp"
#include "types.hpp"

namespace zhetapi {

// Symbol table alias
template <class T>
using symtab = std::unordered_map <std::string, T>;

// Other aliases
using signature = std::vector <std::type_index>;
using overloads = std::vector <std::pair <signature, Token *>>;

// Class for storing overloads
class engine_base {
	symtab <overloads>	_overloads;

	// TODO: this doesn't really need to be Token *
	symtab <Token *>	_universal;

	// Private methods
	std::string gen_overload_msg(const signature &, const std::string &);
	signature gen_signature(const std::vector <Token *> &);
public:
	engine_base();
	engine_base(const engine_base &);

	engine_base &operator=(const engine_base &);

	~engine_base();

	Token *compute(const std::string &, const std::vector <Token *> &);

	bool present(const std::string &) const;

	std::string overload_catalog(const std::string &);
	
	// Exceptions
	class unknown_op_overload : public std::runtime_error {
	public:
		explicit unknown_op_overload(const std::string &str)
				: std::runtime_error(str) {}
	};

	class gensig_nullptr : public std::runtime_error {
	public:
		explicit gensig_nullptr()
				: std::runtime_error("") {}
	};
};

}

#endif
