#ifndef DEFAULTS_H
#define DEFAULTS_H

// C++ Standard Libraries

// Cusotm Built Libraries
#include "token.h"
#include "operand.h"
#include "operation.h"

namespace tokens {
	template <typename data_t>
	class defaults : public token {
			private:
		/* The following are the initializations
		 * of the operations themselves */
                static operation <operand <data_t>> none_op;
		static operation <operand <data_t>> add_op;
		static operation <operand <data_t>> sub_op;
		static operation <operand <data_t>> mult_op;
		static operation <operand <data_t>> div_op;
                static operation <operand <data_t>> exp_op;
		static operation <operand <data_t>> mod_op;

		/* The following are the functions
		 * correspodning to each of the operations */
                static operand <data_t> none_f(const std::vector <operand <data_t>> &);
		static operand <data_t> add_f(const std::vector <operand <data_t>> &);
		static operand <data_t> sub_f(const std::vector <operand <data_t>> &);
		static operand <data_t> mult_f(const std::vector <operand <data_t>> &);
		static operand <data_t> div_f(const std::vector <operand <data_t>> &);
                static operand <data_t> exp_f(const std::vector <operand <data_t>> &);
		static operand <data_t> mod_f(const std::vector <operand <data_t>> &);
	public:
		/* Virtualized functions */
		type caller() const override;
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
		
		static operation <operand <data_t>> opers[];
	};

	/* Corresponding functios */
        template <typename data_t>
	operand <data_t> defaults <data_t> ::none_f(const std::vector <operand <data_t>> &inputs)
	{
		return operand <data_t>();
	}

	template <typename data_t>
	operand <data_t> defaults <data_t> ::add_f(const std::vector <operand <data_t>> &inputs)
	{
		operand <data_t> new_oper_t = operand <data_t>(inputs[0].get() + inputs[1].get());
		return new_oper_t;
	}

        template <typename data_t>
	operand <data_t> defaults <data_t> ::sub_f(const std::vector <operand <data_t>> &inputs)
	{
		operand <data_t> new_oper_t = operand <data_t>(inputs[0].get() - inputs[1].get());
		return new_oper_t;
	}

        template <typename data_t>
	operand <data_t> defaults <data_t> ::mult_f(const std::vector <operand <data_t>> &inputs)
	{
		operand <data_t> new_oper_t = operand <data_t>(inputs[0].get() * inputs[1].get());
		return new_oper_t;
	}

        template <typename data_t>
	operand <data_t> defaults <data_t> ::div_f(const std::vector <operand <data_t>> &inputs)
	{
		operand <data_t> new_oper_t = operand <data_t>(inputs[0].get() / inputs[1].get());
		return new_oper_t;
	}

        template <typename data_t>
	operand <data_t> defaults <data_t> ::exp_f(const std::vector <operand <data_t>> &inputs)
	{
		operand <data_t> new_oper_t = operand <data_t>(std::pow(inputs[0].get(), inputs[1].get()));
		return new_oper_t;
	}

	template <typename data_t>
	operand <data_t> defaults <data_t> ::mod_f(const std::vector <operand <data_t>> &inputs)
	{
		// Need to convert to integer later
		operand <data_t> new_oper_t = operand <data_t>(inputs[0].get() % inputs[1].get());
		return new_oper_t;
	}

	// Defaults's default operations
        template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::none_op = operation <operand <data_t>>
	(std::string {"none_op"}, defaults <data_t> ::none_f, 2, std::vector
        <std::string> {}, operation <operand <data_t>>::NA_L0, std::vector <std::string> {""});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::add_op = operation <operand <data_t>>
	(std::string {"add_op"}, defaults <data_t> ::add_f, 2, std::vector
        <std::string> {"+", "plus", "add"}, operation <operand <data_t>>::SA_L1,
	std::vector <std::string> {"8+8"});

        template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::sub_op = operation <operand <data_t>>
	(std::string {"sub_op"}, defaults <data_t> ::sub_f, 2, std::vector
        <std::string> {"-", "minus"}, operation <operand <data_t>>::SA_L1,
	std::vector <std::string> {"8-8"});

        template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::mult_op = operation <operand <data_t>>
	(std::string {"mult_op"}, defaults <data_t> ::mult_f, 2, std::vector
        <std::string> {"*", "times", "by"}, operation <operand <data_t>>::MDM_L2,
	std::vector <std::string> {"8*8"});

        template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::div_op = operation <operand <data_t>>
	(std::string {"div_op"}, defaults <data_t> ::div_f, 2, std::vector
        <std::string> {"/", "divided by"}, operation <operand <data_t>>::MDM_L2,
	std::vector <std::string> {"8/8"});

        template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::exp_op = operation <operand <data_t>>
	(std::string {"exp_op"}, defaults <data_t> ::exp_f, 2, std::vector
        <std::string> {"^", "to", "to the power of"}, operation <operand <data_t>>::EXP_L3,
	std::vector <std::string> {"8^8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::mod_op = operation <operand <data_t>>
	(std::string {"mod_op"}, defaults <data_t> ::exp_f, 2, std::vector
        <std::string> {"%", "mod"}, operation <operand <data_t>>::MDM_L2,
	std::vector <std::string> {"8%8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::opers[] = {
		add_op, sub_op, mult_op,
                div_op, exp_op, mod_op,
		none_op
	};
	
	typedef defaults <def_t> defaults_t;
}

#endif