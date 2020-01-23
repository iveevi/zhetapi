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
#include "token_tree.h"

namespace tokens {
	// Beginning of the parser class
	template <class data_t>
	class parser : public token {
		/* The following are states of parsing
		 * the expressions, etc. Immediate resolution
		 * is carried out */
		enum STATES {NORM, PAREN};
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
		static std::pair <token *, std::size_t> get_next(std::string,
			std::size_t) noexcept(false);
		static std::vector <token *> get_tokens(std::string);

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
			input, std::size_t index)
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
		
		trees::token_tree <data_t> tr;

                ss.seekg(index);

                for (i = index; i < input.length(); i++) {
                        c = input[i];

			switch(c) {
			case '(':
				state = PAREN;
				break;
			case ')':
				// Throw error
				std::cout << "Error - Missing \
					opening parenthesis" << std::endl;
				return {nullptr, -1};
			}

			if (state == PAREN) {
				// Ignore the parenthesis
				ss.seekg(i + 1);
				while (ss >> c) {
					if (c == ')') {
						// read parenthesis
						ss >> c;
						// std::cout << "paren: " << paren << std::endl;
						tr = trees::token_tree <data_t> (paren);
						return {tr.value()->dptr->get_oper(), ss.tellg()};
					}
					paren += c;
				}
			}

                        if (std::isdigit(c)) {
                                ss >> oper;
                                return {new operand <data_t> (oper),
					ss.tellg()};
                        }

                        // c is an operation
                        // or a grouping term
                        if (!std::isspace(c))
                                cumul += c;
                        
                        opn_index = get_matching(cumul);
                        if (opn_index != defaults <data_t> ::NONOP)
                                return {&defaults <data_t> ::opers[opn_index], i + 1};
                }

                return {nullptr, -1};
        }

	template <typename data_t>
	std::vector <token *> parser <data_t> ::get_tokens(std::string input)
	{
		std::pair <token *, std::size_t> opair;
		std::vector <token *> tokens;
		std::size_t index = 0;
		int ses_len = 5;

		while (true) {
			opair = get_next(input, index);

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

		// stl_reveal <token *> (tokens, [](token *t) {return t->str();});

		// std::cout << "Returning" << std::endl;

		return tokens;
	}

        template <typename data_t>
        std::size_t parser <data_t> ::get_matching(std::string str)
        {
                for (int i = 0; i < defaults <data_t> ::NOPERS; i++) {
                        if (defaults <data_t> ::opers[i].matches(str))
                                return i;
                }

                return defaults <data_t> ::NONOP;
        }

        // Derived member functions
        template <typename data_t>
        token::type parser <data_t> ::caller() const
        {
                return PARSER;
        }

        template <typename data_t>
        std::string parser <data_t> ::str() const
        {
                // Add some more description
                // to the returned string
                return "parser";
        }
}

#endif
