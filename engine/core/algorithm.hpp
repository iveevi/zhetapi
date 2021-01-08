#ifndef ALGORITHM_H_
#define ALGORITHM_H_

// C++ headers
#include <string>
#include <vector>

// Engine headers
#include <token.hpp>

namespace zhetapi {

template <class T, class U>
class Barn;

template <class T, class U>
class algorithm : public Token {
	std::string			__ident;

	std::vector <std::string>	__args;

	// Compile when called on
	// std::vector <std::string>			__statements;

	std::string			__alg;
public:
	algorithm();
	algorithm(std::string, const std::vector <std::string> &,
			const std::string &);
	
	Token *execute(const Barn <T, U> &, const std::vector <Token *> &);
	
	const std::string &symbol() const;

	// Virtual functions
	type caller() const override;
	Token *copy() const override;
	std::string str() const override;

	virtual bool operator==(Token *) const override;
};

// Constructors
template <class T, class U>
algorithm <T, U> ::algorithm() {}

template <class T, class U>
algorithm <T, U> ::algorithm(std::string ident,
		const std::vector <std::string> &args,
		const std::string &alg) : __ident(ident),
		__args(args), __alg(alg) {}

// Executing the function
template <class T, class U>
Token *algorithm <T, U> ::execute(const Barn <T, U> &barn, const std::vector <Token *> &args)
{
	Barn <T, U> cpy(barn);

	// For now, no default arguments or overloads
	assert(args.size() == __args.size());

	std::cout << "BARNS:" << std::endl;

	cpy.print();

	size_t n = __args.size();
	for (size_t i = 0; i < n; i++)
		cpy.put(args[i], __args[i]);

	cpy.print();

	// Return the "return" value instead of nullptr
	return nullptr;
}

// Symbol
template <class T, class U>
const std::string &algorithm <T, U> ::symbol() const
{
	return __ident;
}

// Virtual functions
template <class T, class U>
Token::type algorithm <T, U> ::caller() const
{
	return Token::alg;
}

template <class T, class U>
Token *algorithm <T, U> ::copy() const
{
	std::cout << "COPYING ALGORITHM" << std::endl;
	return new algorithm(__ident, __args, __alg);
}

template <class T, class U>
std::string algorithm <T, U> ::str() const
{
	return __ident;
}

template <class T, class U>
bool algorithm <T, U> ::operator==(Token *tptr) const
{
	algorithm *alg = dynamic_cast <algorithm *> (tptr);

	if (alg == nullptr)
		return false;

	return alg->__ident == __ident;
}

}

#endif