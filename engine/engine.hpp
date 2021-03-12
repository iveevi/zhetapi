#ifndef BARN_H_
#define BARN_H_

// Engine headers
#include <variable.hpp>
#include <function.hpp>
#include <registration.hpp>

#include <core/algorithm.hpp>
#include <core/engine_base.hpp>

namespace zhetapi {

class Engine : public engine_base {
	// Broader scope
	Engine *		__stack = nullptr;

	symtab <Variable>	__var_table;
	symtab <Function>	__ftr_table;

	symtab <Registrable>	__reg_table;
	symtab <algorithm>	__alg_table;

	// Private methods
	void set_origin_stack(Engine *);
public:
	Engine();
	Engine(const Engine &);

	Engine &operator=(const Engine &);

	// Actions
	Engine *pop_stack();

	bool present(const std::string &) const;

	void put(Variable);
	void put(Function);
	void put(Registrable);
	void put(algorithm);

	void put(const std::string &, Token *);

	// Are these necessary?
	Variable &retrieve_variable(const std::string &);
	Function &retrieve_function(const std::string &);

	Token *get(const std::string &);

	void list() const;
	void list_registered(std::string) const;
};

}

#endif
