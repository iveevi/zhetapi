#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C/C++ headers
#include <ostream>
#include <string>

#include <dlfcn.h>

// Engine headers
#include <core/common.hpp>
#include <core/node_manager.hpp>

namespace zhetapi {

class Function : public Token {
	std::string			_symbol;
	std::vector <std::string>	_params;
	node_manager			_manager;
	size_t				_threads;
public:
	Function();
	Function(const char *);
	Function(const std::string &);
	Function(const std::string &, Engine *);

	Function(const std::string &,
		const std::vector <std::string> &,
		const node_manager &);

	Function(const Function &);

	bool is_variable(const std::string &) const;

	std::string &symbol();
	const std::string symbol() const;

	void set_threads(size_t);

	Token *operator()(std::vector <Token *>);

	template <class ... A>
	Token *operator()(A ...);

	template <size_t, class ... A>
	Token *operator()(A ...);

	template <class ... A>
	Token *derivative(const std::string &, A ...);

	Function differentiate(const std::string &) const;

	friend bool operator<(const Function &, const Function &);
	friend bool operator>(const Function &, const Function &);

	// Virtual overloads
	Token::type caller() const override;
	std::string str() const override;
	Token *copy() const override;
	bool operator==(Token *) const override;

	// Printing
	void print() const;

	std::string display() const;

	friend std::ostream &operator<<(std::ostream &, const Function &);
private:
	template <class A>
	void gather(std::vector <Token *> &, A);

	template <class A, class ... B>
	void gather(std::vector <Token *> &, A, B ...);

	size_t index(const std::string &) const;
public:
	// Exception classes
	class invalid_definition {};

	// Static variables
	static Engine engine;

	static double h;
};

template <class ... A>
Token *Function::operator()(A ... args)
{
	std::vector <Token *> tokens;

	gather(tokens, args...);

	assert(tokens.size() == _params.size());

	return _manager.substitute_and_compute(tokens, _threads);
}

// Gathering facilities
template <class A>
void Function::gather(std::vector <Token *> &toks, A in)
{
	toks.push_back(new Operand <A>(in));
}

template <class A, class ... B>
void Function::gather(std::vector <Token *> &toks, A in, B ... args)
{
	toks.push_back(new Operand <A> (in));

	gather(toks, args...);
}

}

#endif
