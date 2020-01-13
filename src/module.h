#ifndef MODULE_H
#define MODULE_H

// C++ Standard Libraries
#include <vector>
#include <sstream>
#include <cmath>

// Custom Built Libraries
#include "token.h"
#include "operand.h"
#include "operation.h"

namespace tokens {
	// Beginning of the module class
	template <class oper_t>
	class module : public token {
	private:
		/* The following are the initializations
		 * of the operations themselves */
                static operation <oper_t> none_op;
		static operation <oper_t> add_op;
		static operation <oper_t> sub_op;
		static operation <oper_t> mult_op;
		static operation <oper_t> div_op;
                static operation <oper_t> exp_op;
		static operation <oper_t> mod_op;

		/* The following are the functions
		 * correspodning to each of the operations */
                static oper_t none_f(const std::vector <oper_t> &);
		static oper_t add_f(const std::vector <oper_t> &);
		static oper_t sub_f(const std::vector <oper_t> &);
		static oper_t mult_f(const std::vector <oper_t> &);
		static oper_t div_f(const std::vector <oper_t> &);
                static oper_t exp_f(const std::vector <oper_t> &);
		static oper_t mod_f(const std::vector <oper_t> &);
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
		
		//static std::vector <token *> *get_tokens(std::string);

                // Always check to make sure
                // oper_t is an operand
                type caller() override;
                std::string str() const override;
		
		/* The following is the array containing
		 * all the default operations, and constants
		 * that represents certain things */
                static const int NONOP = -0x1;
		static const int ADDOP = 0x0;
                static const int SUBOP = 0x1;
                static const int MULTOP = 0x2;
                static const int DIVOP = 0x3;
                static const int EXPOP = 0x4;
		static const int MODOP = 0x5;
                static const int NOPERS = 0x6;
		
		static operation <oper_t> opers[];
	};

	/* Corresponding functios */
        template <typename oper_t>
	oper_t module <oper_t> ::none_f(const std::vector <oper_t> &inputs)
	{
		return oper_t();
	}

	template <typename oper_t>
	oper_t module <oper_t> ::add_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(inputs[0].get() + inputs[1].get());
		return new_oper_t;
	}

        template <typename oper_t>
	oper_t module <oper_t> ::sub_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(inputs[0].get() - inputs[1].get());
		return new_oper_t;
	}

        template <typename oper_t>
	oper_t module <oper_t> ::mult_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(inputs[0].get() * inputs[1].get());
		return new_oper_t;
	}

        template <typename oper_t>
	oper_t module <oper_t> ::div_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(inputs[0].get() / inputs[1].get());
		return new_oper_t;
	}

        template <typename oper_t>
	oper_t module <oper_t> ::exp_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(std::pow(inputs[0].get(), inputs[1].get()));
		return new_oper_t;
	}

	template <typename oper_t>
	oper_t module <oper_t> ::mod_f(const std::vector <oper_t> &inputs)
	{
		// Need to convert to integer later
		oper_t new_oper_t = oper_t(inputs[0].get() % inputs[1].get());
		return new_oper_t;
	}

	// Module's default operations
        template <typename oper_t>
	operation <oper_t> module <oper_t> ::none_op = operation <oper_t>
	(std::string {"none_op"}, module <oper_t> ::none_f, 2, std::vector
        <std::string> {}, operation <oper_t>::NA_L0, std::vector <std::string> {""});

	template <typename oper_t>
	operation <oper_t> module <oper_t> ::add_op = operation <oper_t>
	(std::string {"add_op"}, module <oper_t> ::add_f, 2, std::vector
        <std::string> {"+", "plus", "add"}, operation <oper_t>::SA_L1,
	std::vector <std::string> {"8+8"});

        template <typename oper_t>
	operation <oper_t> module <oper_t> ::sub_op = operation <oper_t>
	(std::string {"sub_op"}, module <oper_t> ::sub_f, 2, std::vector
        <std::string> {"-", "minus"}, operation <oper_t>::SA_L1,
	std::vector <std::string> {"8-8"});

        template <typename oper_t>
	operation <oper_t> module <oper_t> ::mult_op = operation <oper_t>
	(std::string {"mult_op"}, module <oper_t> ::mult_f, 2, std::vector
        <std::string> {"*", "times", "by"}, operation <oper_t>::MDM_L2,
	std::vector <std::string> {"8*8"});

        template <typename oper_t>
	operation <oper_t> module <oper_t> ::div_op = operation <oper_t>
	(std::string {"div_op"}, module <oper_t> ::div_f, 2, std::vector
        <std::string> {"/", "divided by"}, operation <oper_t>::MDM_L2,
	std::vector <std::string> {"8/8"});

        template <typename oper_t>
	operation <oper_t> module <oper_t> ::exp_op = operation <oper_t>
	(std::string {"exp_op"}, module <oper_t> ::exp_f, 2, std::vector
        <std::string> {"^", "to", "to the power of"}, operation <oper_t>::EXP_L3,
	std::vector <std::string> {"8^8"});

	template <typename oper_t>
	operation <oper_t> module <oper_t> ::mod_op = operation <oper_t>
	(std::string {"mod_op"}, module <oper_t> ::exp_f, 2, std::vector
        <std::string> {"%", "mod"}, operation <oper_t>::MDM_L2,
	std::vector <std::string> {"8%8"});

	template <typename oper_t>
	operation <oper_t> module <oper_t> ::opers[] = {
		add_op, sub_op, mult_op,
                div_op, exp_op, mod_op,
		none_op
	};
	
	typedef module <num_t> module_t;

        // Module's parsing functions
        template <typename oper_t>
        std::pair <token *, std::size_t> module <oper_t> ::get_next(std::string
			input, std::size_t index)
        {
                // Add parenthesis (groups)
                // after basic operand parsing
                // is working
                std::size_t i;
                char c;

                // Accumulated string
                std::string cumul;

                std::istringstream ss(input);
                std::size_t opn_index;
                oper_t oper;

                ss.seekg(index);

                for (i = index; i < input.length(); i++) {
                        c = input[i];

                        if (std::isdigit(c)) {
                                ss >> oper;
                                i = ss.tellg();
                                return {new oper_t(oper), i};
                        }

                        // c is an operation
                        // or a grouping term
                        if (!std::isspace(c))
                                cumul += c;
                        
                        opn_index = get_matching(cumul);
                        if (opn_index != NONOP)
                                return {&opers[opn_index], i + 1};
                }

                return {nullptr, -1};
        }

	template <typename oper_t>
	std::vector <token *> module <oper_t> ::get_tokens(std::string input)
	{
		std::pair <token *, std::size_t> opair;
		std::vector <token *> tokens;
		std::size_t index = 0;
		int ses_len = 5;

		while (true) {
			opair = get_next(input, index);

                        if (opair.second == UINT64_MAX) {
                                tokens.push_back(opair.first);
                                break;
                        }

			tokens.push_back(opair.first);
			index = opair.second;
		}

		return tokens;
	}

        template <typename oper_t>
        std::size_t module <oper_t> ::get_matching(std::string str)
        {
                for (int i = 0; i < NOPERS; i++) {
                        if (opers[i].matches(str))
                                return i;
                }

                return NONOP;
        }

        // Derived member functions
        template <typename oper_t>
        token::type module <oper_t> ::caller()
        {
                return MODULE;
        }

        template <typename oper_t>
        std::string module <oper_t> ::str() const
        {
                // Add some more description
                // to the returned string
                return "module";
        }
}

#endif