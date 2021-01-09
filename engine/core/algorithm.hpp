#ifndef ALGORITHM_H_
#define ALGORITHM_H_

// C++ headers
#include <string>
#include <vector>

// Engine headers
#include <core/node_manager.hpp>

namespace zhetapi {

template <class T, class U>
class Barn;

template <class T, class U>
class node_manager;

// Algorithm class
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
	Token *execute(Barn <T, U> &barn, std::string str);

	std::vector <std::string> split(std::string str);
	
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

	size_t n = __args.size();
	for (size_t i = 0; i < n; i++)
		cpy.put(args[i], __args[i]);

	// Use the definition line number
	bool quoted = false;
	int paren = 0;
	
	std::string tmp;

	size_t i = 0;
	
	n = __alg.length();

	char c;
	while ((i < n) && (c = __alg[i++])) {
		if (!quoted) {
			if (c == '\"')
				quoted = true;
			if (c == '(')
				paren++;
			if (c == ')')
				paren--;
			
			if (c == '\n' || (!paren && c == ',')) {
				if (!tmp.empty()) {
					execute(cpy, tmp);

					tmp.clear();
				}
			} else if (!isspace(c)) {
				tmp += c;
			}
		} else {
			if (c == '\"')
				quoted = false;
			
			tmp += c;
		}
	}

	// Return the "return" value instead of nullptr
	return nullptr;
}

template <class T, class U>
Token *algorithm <T, U> ::execute(Barn <T, U> &barn, std::string str)
{
	// Skip comments
	if (str[0] == '#')
		return nullptr;

	std::vector <std::string> tmp = split(str);
	
	size_t tsize = tmp.size();
	if (tsize > 1) {
		zhetapi::Token *tptr = nullptr;
		
		try {
			zhetapi::node_manager <double, int> mg(tmp[tsize - 1], barn);

			tptr = mg.value();
		} catch (...) {}

		for (int i = tsize - 2; i >= 0; i--) {
			std::string ftr = tmp[i] + " = " + tmp[tsize - 1];

			try {
				zhetapi::Function <double, int> f = ftr;

				barn.put(f);
			} catch (...) {
				barn.put(tptr, tmp[i]);
			}
		}
		
		delete tptr;
	} else {		
		// All functions and algorithms are stored in barn
		node_manager <double, int> mg;
		
		try {
			mg = node_manager <double, int> (str, barn);
		} catch (node_manager <double, int> ::undefined_symbol e) {
			std::cout << "Error at line " << 0
				<< ": undefined symbol \""
				<< e.what() << "\"" << std::endl;

			exit(-1);
		}

		/* std::cout << "mg:" << std::endl;
		mg.print(); */

		// "Execute" the statement
		return mg.value(barn);
	}

	return nullptr;
}

// Splitting equalities
template <class T, class U>
std::vector <std::string> algorithm <T, U> ::split(std::string str)
{
	bool quoted = false;

	char pc = 0;

	std::vector <std::string> out;

	size_t n = str.length();

	std::string tmp;
	for (size_t i = 0; i < n; i++) {
		if (!quoted) {
			bool ignore = false;

			if (pc == '>' || pc == '<' || pc == '!'
				|| (i > 0 && str[i - 1] == '='))
				ignore = true;
			
			if (!ignore && str[i] == '=') {
				if (i < n - 1 && str[i + 1] == '=') {
					tmp += "==";
				} else if (!tmp.empty()) {
					out.push_back(tmp);

					tmp.clear();
				}
			} else {
				if (str[i] == '\"')
					quoted = true;
				
				tmp += str[i];
			}
		} else {
			if (str[i] == '\"')
				quoted = false;
			
			tmp += str[i];
		}

		pc = str[i];
	}

	if (!tmp.empty())
		out.push_back(tmp);
	
	/* cout << "split:" << endl;
	for (auto s : out)
		cout << "\ts = " << s << endl; */

	return out;
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
