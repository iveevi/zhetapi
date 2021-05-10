#include <engine.hpp>

#include <core/algorithm.hpp>

#include <lang/compilation.hpp>

namespace zhetapi {

// Constructors
algorithm::algorithm() {}

algorithm::algorithm(const algorithm &other)
		: _ident(other._ident), _alg(other._alg),
		_args(other._args), _compiled(other._compiled) {}

algorithm::algorithm(
		const std::string &ident,
		const std::string &alg,
		const std::vector <std::string> &args)
		: _ident(ident), _args(args),
		_alg(alg) {}

// TODO: remove the alg parameter
algorithm::algorithm(
		const std::string &ident,
		const std::string &alg,
		const std::vector <std::string> &args,
		const node_manager &compiled)
		: _ident(ident), _args(args),
		_alg(alg), _compiled(compiled) {}

void algorithm::compile(Engine *engine)
{
	_compiled = lang::compile_block(engine, _alg, _args, _pardon);
}

// Executing the function
Token *algorithm::execute(Engine *engine, const std::vector <Token *> &args)
{
	// Ignore arguments for now
	if (_compiled.empty())
		compile(engine);
	
	engine = push_and_ret_stack(engine);

	Token *tptr = _compiled.substitute_and_seq_compute(engine, args);

	engine = pop_and_del_stack(engine);

	// Check returns
	if (dynamic_cast <Operand <Token *> *> (tptr))
		return (dynamic_cast <Operand <Token *> *> (tptr))->get();

	return nullptr;
}

// Symbol
const std::string &algorithm::symbol() const
{
	return _ident;
}

void algorithm::print() const
{
	std::cout << "COMPILED:" << std::endl;
	_compiled.print();
}

bool algorithm::empty() const
{
	return _compiled.empty();
}

// Virtual functions
Token::type algorithm::caller() const
{
	return Token::alg;
}

Token *algorithm::copy() const
{
	return new algorithm(_ident, _alg, _args, _compiled);
}

std::string algorithm::dbg_str() const
{
	return "alg-\"" + _ident + "\"";
}

bool algorithm::operator==(Token *tptr) const
{
	algorithm *alg = dynamic_cast <algorithm *> (tptr);

	if (alg == nullptr)
		return false;

	return alg->_ident == _ident;
}

}
