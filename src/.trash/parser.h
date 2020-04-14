#ifndef PARSER_H
#define PARSER_H

// C++ Standard Libraries
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>

// Custom Built Libraries
#include "debug.h"
#include "token.h"
#include "operand.h"
#include "operation.h"
#include "defaults.h"
#include "tree.h"
#include "var_stack.h"

// remove actual parser class

// Beginning of the parser class
// remove derivation to token class
template <class data_t>
class parser : public token {
	/* The following are states of parsing
	 * the expressions, etc. Immediate resolution
	 * is carried out */
	enum STATES {NORM, PAREN, OPERATION};
public:
	/* The following are static member functions that
	 * give purpose to the tokens
	 *
	 * const token &get_next( std::string, std::size_t):
	 *   returns the next valid token in the passed
	 *   string from the specified index, or throws an
	 *   error if no token was detected and modifies the
	 *   passed index value appropriately
	 */

	// keep reference to stack variable, as
	// the stack changes fromfind queries
	static std::pair <token *, std::size_t> get_next(std::string,
		std::size_t, var_stack <data_t> &) noexcept(false);
	
	static std::vector <token *> get_tokens(std::string,
			var_stack <data_t> &);

	// Returns the index of the operation who's format
	// matches the format this passed, and none_op if none
	static std::size_t get_matching(std::string);

	// Always check to make sure
	// operand <data_t> is an operand
	type caller() const override;
	std::string str() const override;
};



// Parser's parsing functions
template <typename data_t>
std::pair <token *, std::size_t> parser <data_t> ::get_next(std::string
		input, std::size_t index, var_stack <data_t> &vst)
{
	// Current state of parsing
	STATES state = NORM;

	// Add parenthesis (groups)
	// after basic operand parsing
	// is working
	std::size_t i;
	char c;

	// Accumulated string
	std::string cumul;
	std::string paren;

	std::istringstream ss(input);
	std::size_t opn_index;

	operand <data_t> oper;
	
	tree <data_t> tr;

	ss.seekg(index);

	for (i = index; i < input.length(); i++) {
		c = input[i];

		switch(c) {
		case '(':
			state = PAREN;
			break;
		case ')':
			// Throw error
			std::cout << "Error - Missing "
				<< "opening parenthesis" << std::endl;
			return {nullptr, -1};
		}

		if (state == PAREN) {
			// Ignore the parenthesis
			// ss.seekg(i + 1);
			for (int si = i + 1; si < input.length(); si++) {
				if (input[si] == ')') {
					/*dp_var(paren);
					dp_msg("IMPORTANT---")
					dp_var(si)
					dp_var(input.length());*/
					// dp_var(ss.tellg());
					// std::cout << "paren: " << paren << std::endl;
					// std::cout << "tellg(): " << ss.tellg() << std::endl;
					tr = tree <data_t> (paren);

					if (si >= input.length() - 1)
						return {new operand <data_t> (tr.value()),
							-1};
					return {new operand <data_t> (tr.value()),
						si + 1};
				}
				paren += input[si];
			}

			// Add auto completion later
			// assuming parenthesis
		}

		if (std::isdigit(c)) {
			// dp_var(c)
			// dp_var(ss.tellg())
			ss >> oper;
			return {new operand <data_t> (oper),
				ss.tellg()};
		}

		// isspace is ok because we already
		// took care of parenthesis (add
		// function sensetivity to parentheses)
		// if (state == OPERATION && isspace(c)) {
		// 	state = NORM;

			
		// }

		// c is the start of an
		// operation or function
		if (!std::isspace(c)) {
			//state = OPERATION;
			cumul += c;
		}

		// dp_msg("got here")
		opn_index = get_matching(cumul);
		// dp_msg("lost after")
		
		if (opn_index != defaults <data_t> ::NONOP) {
			return {&defaults <data_t> ::opers[opn_index], i + 1};
		} else if (!cumul.empty()) {
			// If not an operation, could be variable
			// std::cout << "Trying to find: " << cumul << std::endl;
			variable <data_t> var;
			try {
				var = vst.find(cumul);
			} catch (typename var_stack <data_t> ::nfe e) {
				// Skip return
				// dp_msg("Her");
				continue;
			}

			// return value of var
			if (i >= input.length() - 1)
				return {new operand <data_t> (var.get()), -1};
			return {new operand <data_t> (var.get()), i + 1};
		}

	}

	return {nullptr, -1};
}

template <typename data_t>
std::vector <token *> parser <data_t> ::get_tokens(std::string input,
		var_stack <data_t> &vst)
{
	std::pair <token *, std::size_t> opair;
	std::vector <token *> tokens;
	std::size_t index = 0;
	int ses_len = 5;
	
	token *t;

	/*std::cout << std::string(10, '*') << std::endl;
	for (int i = 0; i < input.length(); i++)
		std::cout << i << "\t" << input[i] << std::endl;
	std::cout << std::endl;*/
	// dp_var(input)

	while (true) {
		opair = get_next(input, index, vst);

		// dp_var(opair.first->str());
		// dp_var(opair.second)

		if (opair.second == UINT64_MAX) {
			// dp_var(input);
			// dp_var(opair.first);
			// dp_var(opair.second);
			tokens.push_back(opair.first);
			break;
		}

		tokens.push_back(opair.first);
		index = opair.second;
	}

	/* stl_reveal(t, tokens, [](token *t) {
		if (t == nullptr)
			return std::string("nullptr");
		return t->str();
	});

	std::cout << "Returning" << std::endl; */
	// std::cout << std::string(5, '*') << std::endl;

	return tokens;
}

template <class T>
const std::pair <token *, size_t> &functor <T> ::get_next
	(const std::string &str, const size_t &index,
	m_state &state, const param_list &m_params, map &m_map,
	var_stack <T> &vst)
{
	auto null = defaults <T> ::opers[defaults <T> ::NOPERS];
	auto opers = defaults <T> ::opers;

	size_t m_size = defaults <T> ::m_size;

	std::pair <token *, size_t> *pr = new
		std::pair <token *, size_t> {nullptr, -1};

	std::istringstream ss(str);
	std::string paren;
	std::string accum;
	size_t i;

	T val;

	ss.seekg(index);
	for (i = index; i < str.length(); i++) {
		if (str[i] == '(') {
			for (size_t j = i + 1; j < str.length(); j++) {
				if (str[j] == ')') {
					node *ptree = build(paren, m_params, m_map);

					pr->first = new group(ptree);
					// print(ptree, 0, 0);
					pr->second = j + 1;
					state = state_operand;

					return *pr;
				}

				paren += str[j];
			}
		} else if (str[i] == ')') {
			throw syntax_error();
		}

		if (isdigit(str[i])) {
			ss >> val;

			pr->first = new operand <T> (val);
			pr->second = ss.tellg();
			state = state_operand;

			break;
		}

		if (!isspace(str[i]))
			accum += str[i];

		if (accum.empty())
			continue;

		// Priority with variables over var-stack vars
		auto var = find_if(m_params.begin(), m_params.end(),
			[&](const variable <T> &v) {
				return v.symbol() == accum;
			});
		
		if (var != m_params.end()) {
			pr->first = new variable <T> (var->symbol(), true);
			pr->second = i + 1;
			state = state_operand;

			break;
		}

		auto itr = std::find_if(opers, opers + m_size,
			[&](const operation <operand <T>> &opn) {
				return opn.matches(accum);
			});
		
		if (itr != nullptr && *itr != null) {
			pr->first = itr;
			pr->second = i + 1;
			state = state_operation;

			break;
		}
	}

	return *pr;
}

template <typename data_t>
std::size_t parser <data_t> ::get_matching(std::string str)
{
	// dp_var(str)
	// dp_msg("in matching process")
	for (int i = 0; i < defaults <data_t> ::NOPERS; i++) {
		// dp_msg("looping")
		//dp_var(i)
		//dp_var(defaults <data_t> ::opers[i].str())
		if (defaults <data_t> ::opers[i].matches(str))
			return i;
	}

	// dp_msg("returning")

	return defaults <data_t> ::NONOP;
}

// Derived member functions
// remove
template <typename data_t>
token::type parser <data_t> ::caller() const
{
	return PARSER;
}

// remove
template <typename data_t>
std::string parser <data_t> ::str() const
{
	// Add some more description
	// to the returned string
	return "parser";
}

#endif
