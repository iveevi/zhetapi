#include <cmath>

#include "../engine/engine.hpp"
#include "../engine/core/algorithm.hpp"
#include "../engine/function.hpp"
#include "../interpreter/builtin/builtin.hpp"

namespace zhetapi {

Engine::Engine(bool defaults)
{
	if (defaults) {
		// Default present
		put("true", new OpB(true));
		put("false", new OpB(false));
		put("null", new Operand <Token *> (nullptr));	// TODO: typedef for this
		put("i", new OpCmpQ(CmpQ(0, 1ll)));

		put("print", new Registrable("print", &bt_print));
		put("println", new Registrable("println", &bt_println));
		put("range", new Registrable("range", &bt_range));
		put("dict", new Registrable("dict", &bt_dict));
	}
}

Engine::Engine(const Engine &other)
		: _stack(other._stack),
		_var_table(other._var_table) {}

Engine &Engine::operator=(const Engine &other)
{
	if (this != &other) {
		_stack = other._stack;
		_var_table = other._var_table;
	}

	return *this;
}

Engine::~Engine()
{
	/* for (auto pair : _var_table)
		delete pair.second; */
}

// Methods
Args Engine::symbol_list() const
{
	Args syms;
	for (const auto &p : _var_table)
		syms.push_back(p.first);

	return syms;
}

// Actions
Engine *Engine::new_stack()
{
	Engine *engine = new Engine();

	engine->set_origin_stack(this);

	return engine;
}

Engine *Engine::get_stack()
{
	return _stack;
}

// Private methods
void Engine::set_origin_stack(Engine *engine)
{
	_stack = engine;
}

// Just copy the token address-wise
void Engine::put(const std::string &str, Token *tptr)
{
	// Check broader scope first
	if (_stack && _stack->get(str)) {
		_stack->put(str, tptr);

		return;
	}

	// Insert appropriately
	if (_var_table.count(str))
		_var_table[str] = tptr;
	else
		_var_table.insert(std::make_pair(str, tptr->copy()));
}

Token *Engine::get(const std::string &str)
{
	// Return modifiable references (TODO: do the same with others)
	if (_var_table.count(str))
		return _var_table[str];
	
	// Check higher stacks
	if (_stack)
		return _stack->get(str);

	return nullptr;
}

void Engine::list() const
{
	std::cout << "\tVariables:" << std::endl;
	for (auto spr : _var_table) {
		std::cout << "\t\t" << spr.first << " ["
			<< spr.second->dbg_str() << "]" << std::endl;
	}
}

// TODO: this is useless
void Engine::list_registered(std::string file) const
{
	printf("Symbols recorded in %s:\n", file.c_str());
	for (auto spr : _var_table)
		std::cout << "\t" << spr.second->dbg_str() << std::endl;
}

// Non-member functions
Engine *push_and_ret_stack(Engine *engine)
{
	return engine->new_stack();
}

Engine *pop_and_del_stack(Engine *engine)
{
	Engine *popped = engine->get_stack();

	delete engine;

	return popped;
}

}
