#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C++ Standard Libraries
#include <string>
#include <vector>
#include <unordered_map>

// Custom Built Libraries
#include "token.h"
#include "variable.h"
#include "constants.h"
#include "operand.h"
#include "defaults.h"

// remove after fixes
#include "debug.h"

/**
 * @tparam T fdsf
 */
template <class T>
class functor {
public:
	struct node {
		token *tok;
		std::vector <node *> leaves;
	};

	class invalid_definition {};
	class syntax_error {};
protected:
	/** functor symbol/name */
	std::string m_name;

	/** functors inputs/parameters,
	used really only for checking correctness */
	std::vector <variable <T>> m_params;

	/** functor tree */
	node *m_root;
public:
	functor(const std::string &);
	functor(const std::string &,
		const std::vector <std::string> &,
		const std::string &);

	void print() const;
protected:
	void print(node *, int, int) const;

	node *build(const std::string &);
	node *build(const std::vector <token *> &);

	static const std::vector <token *> symbols
		(const std::string &);

	static const std::pair <token *, size_t>
		&next(const std::string &, const size_t &);
	
	/* replace with vector find
	static size_t get_matching(std::string); */
};

/**
 * @brief A constructor which takes in
 * 	the complete definition of the functor
 * 	eg. "f(x) = x^2"
 * 
 * @tparam The base type, with which calculations
 * 	are performed under, eg. int and double
 *
 * @param in The complete definition of the
 *	functor as a string, mentioned @brief
 */
template <class T>
functor <T> ::functor(const std::string &in)
{
	std::vector <std::string> params;
	std::string name;
	std::string expr;

	size_t count = 0;
	size_t index;
	for (int i = 0; i < in.length(); i++) {
		if (in[i] == '=') {
			index = i;
			count++;
		}
	}

	if (count != ASSIGNMENT)
		throw invalid_definition();

	name = in.substr(0, index);
	expr = in.substr(index + 1);

	dp_var(name);
	dp_var(expr);

	size_t start = -1;
	size_t end;
	for (int i = 0; i < name.length(); i++) {
		if (in[i] == '(' && start == -1)
			start = i;
		if (in[i] == ')')
			end = i;
	}

	std::string vars = name.substr(start + 1, end - start - 1);
	name = name.substr(0, start);

	dp_var(vars);
	dp_var(name);

	std::string var;
	for (int i = 0; i < vars.length(); i++) {
		if (vars[i] == ',' && !var.empty()) {
			dp_var(var);
			params.push_back(var);
			var.clear();
		} else if (!isspace(vars[i])) {
			var += vars[i];
		}
	}

	if (!var.empty()) {
		dp_var(var);
		params.push_back(var);
	}

	m_name = name;

	for (std::string str : params)
		m_params.push_back(variable <T> (str, true));
	
	m_root = build(expr);
	print();
}

template <class T>
typename functor <T> ::node *functor <T> ::build
	(const std::string &str)
{
	return build(symbols(str));
}

template <class T>
typename functor <T> ::node *functor <T> ::build
	(const std::vector <token *> &toks)
{

}

template <class T>
const std::vector <token *> functor <T> ::symbols(const std::string &str)
{
	std::pair <token *, size_t> pr;
	std::vector <token *> toks;
	size_t index = 0;
	
	while (true) {
		pr = next(str, index);

		toks.push_back(pr.first);
		index = pr.second;

		if (index == (size_t) -1)
			break;
	}

	return toks;
}

template <class T>
const std::pair <token *, size_t> &functor <T> ::next
	(const std::string &str, const size_t &index)
{
	using defaults <T> ::opers;
	using defaults <T> ::m_size;

	std::istringstream ss(str);
	std::string paren;
	std::string accum;
	size_t i;

	T val;

	for (i = index; i < str.length(); i++) {
		/* handle parenthesis later
		if (str[i] == '(') {
			for (size_t j = i + 1; j < str.length(); j++) {
				if (str[j] == ')') {

				}
			}
		} else if (str[i] == ')') {
			throw syntax_error();
		} */

		if (isdigit(str[i])) {
			ss >> val;
			return {new operand <T> (val), ss.tellg()};
		}

		if (!isspace(str[i]))
			accum += str[i];

		auto itr = find(opers, opers + m_size,
			[&](const operation <operand <T>> &opn) {
				return opn.matches(accum);
			});
	}
}

// 'Debugging' functors
//
//

template <class T>
void functor <T> ::print() const
{
	std::cout << "PRINTING TREE" << std::endl;
	print(m_root, 1, 0);
}

template <class T>
void functor <T> ::print(node *nd, int num,
	int lev) const
{
	if (nd == nullptr) 
		return;

	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	std::cout << "#" << num << " - " << nd << std::endl;

	counter = 0;
	for (auto it = nd->leaves.begin(); it != nd->leaves.end(); it++, counter++)
		print(*it, counter + 1, lev + 1);
}

/** functor evaluation map
std::unordered_map <std::string, T> m_vmap; */

#endif
