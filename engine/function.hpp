#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C/C++ headers
#include <ostream>
#include <string>

#include <dlfcn.h>

// Engine headers
#include <core/node_manager.hpp>

// Undefine compilation flag
#ifdef ZHP_FUNCTION_COMPILE_GENERAL

#undef ZHP_FUNCTION_COMPILE_GENERAL

#endif

// Set the correct compilation mode
#ifdef __linux__

#define ZHP_FUNCTION_COMPILE_GENERAL
#define ZHP_COMPILE_LINUX

#endif

namespace zhetapi {

class Function : public Token {
	std::string			__symbol;
	std::vector <std::string>	__params;
	node_manager			__manager;
	size_t				__threads;
public:
	Function();
	Function(const char *);
	Function(const std::string &);
	Function(const std::string &, Barn *);

	Function(const std::string &,
		const std::vector <std::string> &,
		const node_manager &);

	Function(const Function &);

	std::string &symbol();
	const std::string symbol() const;

	void set_threads(size_t);

	Token *operator()(std::vector <Token *>);

	template <class ... A>
	Token *operator()(A ...);

	template <size_t, class ... A>
	Token *operator()(A ...);

	template <class ... A>
	Token *derivative(const ::std::string &, A ...);

	Function differentiate(const ::std::string &) const;

	friend bool operator<(const Function &, const Function &);
	friend bool operator>(const Function &, const Function &);

	std::string generate_general() const;

	void *compile_general() const;

	// Virtual overloads
	Token::type caller() const override;
	std::string str() const override;
	Token *copy() const override;
	bool operator==(Token *) const override;

	// Printing
	void print() const;

	std::string display() const;

	friend std::ostream &operator<<(::std::ostream &, const Function &);
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
	static Barn barn;

	static double h;
};

template <class ... A>
Token *Function::operator()(A ... args)
{
	std::vector <Token *> Tokens;

	gather(Tokens, args...);

	assert(Tokens.size() == __params.size());

	return __manager.substitute_and_compute(Tokens, __threads);
}

// Gathering facilities
template <class A>
void Function::gather(::std::vector <Token *> &Tokens, A in)
{
	Tokens.push_back(new Operand <A>(in));
}

template <class A, class ... B>
void Function::gather(std::vector <Token *> &Tokens, A in, B ... args)
{
	Tokens.push_back(new Operand <A>(in));

	gather(Tokens, args...);
}

}

#endif
