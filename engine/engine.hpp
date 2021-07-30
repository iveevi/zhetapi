#ifndef ENGINE_H_
#define ENGINE_H_

// C/C++ headers
#include <unordered_map>

// Engine headers
#include "function.hpp"
#include "registration.hpp"

#include "core/algorithm.hpp"
#include "core/common.hpp"

namespace zhetapi {

// Aliases
template <class T>
using Symtab = std::unordered_map <std::string, T>;

class Engine {
	// Broader scope
	Engine *		_stack = nullptr;
	
	Symtab <Token *>	_var_table;

	// Private methods
	void set_origin_stack(Engine *);
public:
	Engine(bool = false);
	Engine(const Engine &);

	Engine &operator=(const Engine &);

	~Engine();

	// List all symbols
	Args symbol_list() const;

	// Actions
	Engine *new_stack();
	Engine *get_stack();

	void put(const std::string &, Token *);

	Token *get(const std::string &);

	void list() const;
	void list_registered(std::string) const;
};

Engine *push_and_ret_stack(Engine *);
Engine *pop_and_del_stack(Engine *);

}

#endif
