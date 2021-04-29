#ifndef BARN_H_
#define BARN_H_

// Engine headers
#include <function.hpp>
#include <registration.hpp>

#include <core/algorithm.hpp>
#include <core/engine_base.hpp>
#include <core/common.hpp>

namespace zhetapi {

struct NULLPTR {};

class Engine : public engine_base {
	// Broader scope
	Engine *		_stack = nullptr;

	symtab <algorithm>	_alg_table;
	symtab <Function>	_ftr_table;
	symtab <Registrable>	_reg_table;
	symtab <Token *>	_var_table;

	// Private methods
	void set_origin_stack(Engine *);
public:
	Engine(bool = false);
	Engine(const Engine &);

	Engine &operator=(const Engine &);

	// List all symbols
	Args symbol_list() const;

	// Actions
	Engine *new_stack();
	Engine *get_stack();

	void put(Function);
	void put(Registrable);
	void put(algorithm);
	void put(const std::string &, Token *);

	// Are these necessary?
	Function &retrieve_function(const std::string &);

	Token *get(const std::string &);

	void list() const;
	void list_registered(std::string) const;
};

Engine *push_and_ret_stack(Engine *);
Engine *pop_and_del_stack(Engine *);

}

#endif
