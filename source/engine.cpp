#include <engine.hpp>
#include <core/algorithm.hpp>
#include <function.hpp>

#include <cmath>

namespace zhetapi {

Engine::Engine() : engine_base() {}

Engine::Engine(const Engine &other)
		: engine_base(other),
		_stack(other._stack),
		_reg_table(other._reg_table),
		_alg_table(other._alg_table),
		_ftr_table(other._ftr_table),
		_var_table(other._var_table) {}

Engine &Engine::operator=(const Engine &other)
{
	if (this != &other) {
		engine_base::operator=(other);

		_stack = other._stack;
		_var_table = other._var_table;
		_ftr_table = other._ftr_table;
		_reg_table = other._reg_table;
		_alg_table = other._alg_table;
	}

	return *this;
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

void Engine::put(Variable var)
{
	if (_var_table.count(var.symbol()))
		_var_table[var.symbol()] = var;
	else
		_var_table.insert(std::make_pair(var.symbol(), var));
}

void Engine::put(Function ftr)
{
	if (_ftr_table.count(ftr.symbol()))
		_ftr_table[ftr.symbol()] = ftr;
	else
		_ftr_table.insert(std::make_pair(ftr.symbol(), ftr));
}

void Engine::put(Registrable reg)
{
	if (_reg_table.find(reg.str()) != _reg_table.end())
		_reg_table[reg.str()] = reg;
	else
		_reg_table.insert(std::make_pair(reg.str(), reg));
}

void Engine::put(algorithm alg)
{
	if (_alg_table.find(alg.symbol()) != _alg_table.end())
		_alg_table[alg.symbol()] = alg;
	else
		_alg_table.insert(std::make_pair(alg.symbol(), alg));
}

void Engine::put(const std::string &str, Token *tptr)
{
	// TODO: should this be so?
	//
	// "Leave the passed pointer alone, and copy it instead"
	put(Variable(tptr->copy(), str));
}

Variable &Engine::retrieve_variable(const std::string &str)
{
	return _var_table[str];
}

Function &Engine::retrieve_function(const std::string &str)
{
	return _ftr_table[str];
}

Token *Engine::get(const std::string &str)
{
	// Prioritize algorithms
	if (_alg_table.count(str))
		return _alg_table[str].copy();
	
	if (_reg_table.count(str))
		return _reg_table[str].copy();
	
	if (_var_table.count(str))
		return _var_table[str].copy();
	
	if (_ftr_table.count(str))
		return _ftr_table[str].copy();
	
	if (_stack)
		return _stack->get(str);
	
	return nullptr;
}

void Engine::list() const
{
	std::cout << "\tVariables:" << std::endl;
	for (auto spr : _var_table)
		std::cout << "\t\t" << spr.second.str() << std::endl;
	
	std::cout << "\tFunctions:" << std::endl;
	for (auto spr : _ftr_table)
		std::cout << "\t\t" << spr.second.str() << std::endl;
}

void Engine::list_registered(std::string file) const
{
	printf("Symbols recorded in %s:\n", file.c_str());
	for (auto spr : _reg_table)
		std::cout << "\t" << spr.second.str() << std::endl;
	for (auto spr : _var_table)
		std::cout << "\t" << spr.second.str() << std::endl;
	for (auto spr : _ftr_table)
		std::cout << "\t" << spr.second.str() << std::endl;
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
