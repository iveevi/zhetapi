#include <engine.hpp>
#include <core/algorithm.hpp>
#include <function.hpp>

#include <cmath>

namespace zhetapi {

Engine::Engine() : engine_base() {}

Engine::Engine(const Engine &other)
		: engine_base(other),
		__stack(other.__stack),
		__reg_table(other.__reg_table),
		__alg_table(other.__alg_table),
		__ftr_table(other.__ftr_table),
		__var_table(other.__var_table) {}

Engine &Engine::operator=(const Engine &other)
{
	if (this != &other) {
		engine_base::operator=(other);

		__stack = other.__stack;
		__var_table = other.__var_table;
		__ftr_table = other.__ftr_table;
		__reg_table = other.__reg_table;
		__alg_table = other.__alg_table;
	}

	return *this;
}

// Actions
Engine *Engine::pop_stack()
{
	Engine *engine = new Engine();

	engine->set_origin_stack(this);

	return engine;
}

// Private methods
void Engine::set_origin_stack(Engine *engine)
{
	__stack = engine;
}

void Engine::put(Variable var)
{
	if (__var_table.count(var.symbol()))
		__var_table[var.symbol()] = var;
	else
		__var_table.insert(std::make_pair(var.symbol(), var));
}

void Engine::put(Function ftr)
{
	if (__ftr_table.count(ftr.symbol()))
		__ftr_table[ftr.symbol()] = ftr;
	else
		__ftr_table.insert(std::make_pair(ftr.symbol(), ftr));
}

void Engine::put(Registrable reg)
{
	if (__reg_table.find(reg.str()) != __reg_table.end())
		__reg_table[reg.str()] = reg;
	else
		__reg_table.insert(std::make_pair(reg.str(), reg));
}

void Engine::put(algorithm alg)
{
	if (__alg_table.find(alg.symbol()) != __alg_table.end())
		__alg_table[alg.symbol()] = alg;
	else
		__alg_table.insert(std::make_pair(alg.symbol(), alg));
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
	return __var_table[str];
}

Function &Engine::retrieve_function(const std::string &str)
{
	return __ftr_table[str];
}

Token *Engine::get(const std::string &str)
{
	// Prioritize algorithms
	if (__alg_table.count(str))
		return __alg_table[str].copy();
	
	if (__reg_table.count(str))
		return __reg_table[str].copy();
	
	if (__var_table.count(str))
		return __var_table[str].copy();
	
	if (__ftr_table.count(str))
		return __ftr_table[str].copy();
	
	return nullptr;
}

void Engine::list() const
{
	std::cout << "\tVariables:" << std::endl;
	for (auto spr : __var_table)
		std::cout << "\t\t" << spr.second.str() << std::endl;
	
	std::cout << "\tFunctions:" << std::endl;
	for (auto spr : __ftr_table)
		std::cout << "\t\t" << spr.second.str() << std::endl;
}

void Engine::list_registered(std::string file) const
{
	printf("Symbols recorded in %s:\n", file.c_str());
	for (auto spr : __reg_table)
		std::cout << "\t" << spr.second.str() << std::endl;
	for (auto spr : __var_table)
		std::cout << "\t" << spr.second.str() << std::endl;
	for (auto spr : __ftr_table)
		std::cout << "\t" << spr.second.str() << std::endl;
}

}
