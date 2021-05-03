#include <engine.hpp>
#include <core/algorithm.hpp>
#include <function.hpp>

#include <cmath>

#include <builtin/basic_io.hpp>

namespace zhetapi {

Engine::Engine(bool defaults) : engine_base()
{
	if (defaults) {
		// Default present
		put("true", new OpB(true));
		put("false", new OpB(false));
		put("null", new Operand <Token *> (nullptr));	// TODO: typedef for this
		put("i", new OpCmpQ(CmpQ(0, 1ll)));

		put(Registrable("print", &bt_print));
		put(Registrable("println", &bt_println));
	}
}

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

// Methods
Args Engine::symbol_list() const
{
	Args syms;

	for (const auto &p : _alg_table)
		syms.push_back(p.first);
	for (const auto &p : _ftr_table)
		syms.push_back(p.first);
	for (const auto &p : _reg_table)
		syms.push_back(p.first);
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
		_var_table[str] = tptr->copy();
	else
		_var_table.insert(std::make_pair(str, tptr->copy()));
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
	if (_reg_table.find(reg.dbg_str()) != _reg_table.end())
		_reg_table[reg.dbg_str()] = reg;
	else
		_reg_table.insert(std::make_pair(reg.dbg_str(), reg));
}

void Engine::put(algorithm alg)
{
	if (_alg_table.find(alg.symbol()) != _alg_table.end())
		_alg_table[alg.symbol()] = alg;
	else
		_alg_table.insert(std::make_pair(alg.symbol(), alg));
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
		return _var_table[str]->copy();
	
	if (_ftr_table.count(str))
		return _ftr_table[str].copy();
	
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
	
	std::cout << "\tFunctions:" << std::endl;
	for (auto spr : _ftr_table)
		std::cout << "\t\t" << spr.second.dbg_str() << std::endl;
	
        std::cout << "\tRegistrables:" << std::endl;
	for (auto spr : _reg_table)
		std::cout << "\t\t" << spr.second.dbg_str() << std::endl;
	
        std::cout << "\tAlgorithms:" << std::endl;
	for (auto spr : _alg_table)
		std::cout << "\t\t" << spr.second.dbg_str() << std::endl;
}

void Engine::list_registered(std::string file) const
{
	printf("Symbols recorded in %s:\n", file.c_str());
	for (auto spr : _reg_table)
		std::cout << "\t" << spr.second.dbg_str() << std::endl;
	for (auto spr : _var_table)
		std::cout << "\t" << spr.second->dbg_str() << std::endl;
	for (auto spr : _ftr_table)
		std::cout << "\t" << spr.second.dbg_str() << std::endl;
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
