#ifndef DEFAULTS_H
#define DEFAULTS_H

// C++ Standard Libraries
#include <cmath>

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
		static operation <operand <data_t>> sin_op;
		static operation <operand <data_t>> cos_op;
		static operation <operand <data_t>> tan_op;
		static operation <operand <data_t>> csc_op;
		static operation <operand <data_t>> sec_op;
		static operation <operand <data_t>> cot_op;

		/* The following are the functions
		 * correspodning to each of the operations */
                static operand <data_t> none_f(const std::vector <operand <data_t>> &);
		static operand <data_t> add_f(const std::vector <operand <data_t>> &);
		static operand <data_t> sub_f(const std::vector <operand <data_t>> &);
		static operand <data_t> mult_f(const std::vector <operand <data_t>> &);
		static operand <data_t> div_f(const std::vector <operand <data_t>> &);
                static operand <data_t> exp_f(const std::vector <operand <data_t>> &);
		static operand <data_t> mod_f(const std::vector <operand <data_t>> &);
		static operand <data_t> sin_f(const std::vector <operand <data_t>> &);
		static operand <data_t> cos_f(const std::vector <operand <data_t>> &);
		static operand <data_t> tan_f(const std::vector <operand <data_t>> &);
		static operand <data_t> csc_f(const std::vector <operand <data_t>> &);
		static operand <data_t> sec_f(const std::vector <operand <data_t>> &);
		static operand <data_t> cot_f(const std::vector <operand <data_t>> &);
	public:
		/* Virtualized functions */
		type caller() const override;
                std::string str() const override;

		/* The following is the array containing
		 * all the default operations, and constants
		 * that represents certain things */
                static const int NONOP = -1;
		static const int ADDOP = 0;
                static const int SUBOP = 1;
                static const int MULTOP = 2;
                static const int DIVOP = 3;
                static const int EXPOP = 4;
		static const int MODOP = 5;
		static const int SINOP = 6;
		static const int COSOP = 7;
		static const int TANOP = 8;
		static const int CSCOP = 9;
		static const int SECOP = 10;
		static const int COTOP = 11;
                static const int NOPERS = 12;
		
		static operation <operand <data_t>> opers[];
	};

	// Virtualized functions
	template <typename data_t>
	token::type defaults <data_t> ::caller() const
	{
		return PARSER;
	}

	template <typename data_t>
	std::string defaults <data_t> ::str() const
	{
		return "defaults";
	}

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

	template <typename data_t>
	operand <data_t> defaults <data_t> ::sin_f(const std::vector <operand <data_t>> &inputs)
	{
		// Need to convert to integer later
		operand <data_t> new_oper_t = operand <data_t>(sin(inputs[0].get()));
		return new_oper_t;
	}

	template <typename data_t>
	operand <data_t> defaults <data_t> ::cos_f(const std::vector <operand <data_t>> &inputs)
	{
		// Need to convert to integer later
		operand <data_t> new_oper_t = operand <data_t>(cos(inputs[0].get()));
		return new_oper_t;
	}

	template <typename data_t>
	operand <data_t> defaults <data_t> ::tan_f(const std::vector <operand <data_t>> &inputs)
	{
		// Need to convert to integer later
		operand <data_t> new_oper_t = operand <data_t>(tan(inputs[0].get()));
		return new_oper_t;
	}

	template <typename data_t>
	operand <data_t> defaults <data_t> ::csc_f(const std::vector <operand <data_t>> &inputs)
	{
		// Need to convert to integer later
		operand <data_t> new_oper_t = operand <data_t>(1.0 / sin(inputs[0].get()));
		return new_oper_t;
	}

	template <typename data_t>
	operand <data_t> defaults <data_t> ::sec_f(const std::vector <operand <data_t>> &inputs)
	{
		// Need to convert to integer later
		operand <data_t> new_oper_t = operand <data_t>(1.0 / cos (inputs[0].get()));
		return new_oper_t;
	}

	template <typename data_t>
	operand <data_t> defaults <data_t> ::cot_f(const std::vector <operand <data_t>> &inputs)
	{
		// Need to convert to integer later
		operand <data_t> new_oper_t = operand <data_t>(1.0 / tan(inputs[0].get()));
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

	// needs to be changed
	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::mod_op = operation <operand <data_t>>
	(std::string {"mod_op"}, defaults <data_t> ::exp_f, 2, std::vector
        <std::string> {"%", "mod"}, operation <operand <data_t>>::MDM_L2,
	std::vector <std::string> {"8%8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::sin_op = operation <operand <data_t>>
	(std::string {"sin_op"}, defaults <data_t> ::sin_f, 1, std::vector
        <std::string> {"sin"}, operation <operand <data_t>>::FUNC_LMAX,
	std::vector <std::string> {"sin 8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::cos_op = operation <operand <data_t>>
	(std::string {"cos_op"}, defaults <data_t> ::cos_f, 1, std::vector
        <std::string> {"cos"}, operation <operand <data_t>>::FUNC_LMAX,
	std::vector <std::string> {"cos 8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::tan_op = operation <operand <data_t>>
	(std::string {"tan_op"}, defaults <data_t> ::tan_f, 1, std::vector
        <std::string> {"tan"}, operation <operand <data_t>>::FUNC_LMAX,
	std::vector <std::string> {"tan 8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::csc_op = operation <operand <data_t>>
	(std::string {"csc_op"}, defaults <data_t> ::csc_f, 1, std::vector
        <std::string> {"csc"}, operation <operand <data_t>>::FUNC_LMAX,
	std::vector <std::string> {"csc 8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::sec_op = operation <operand <data_t>>
	(std::string {"sec_op"}, defaults <data_t> ::sec_f, 1, std::vector
        <std::string> {"sec"}, operation <operand <data_t>>::FUNC_LMAX,
	std::vector <std::string> {"sec 8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::cot_op = operation <operand <data_t>>
	(std::string {"cot_op"}, defaults <data_t> ::cot_f, 1, std::vector
        <std::string> {"cot"}, operation <operand <data_t>>::FUNC_LMAX,
	std::vector <std::string> {"cot 8"});

	template <typename data_t>
	operation <operand <data_t>> defaults <data_t> ::opers[] = {
		add_op, sub_op, mult_op,
                div_op, exp_op, mod_op,
		sin_op, cos_op, tan_op,
		csc_op, sec_op, cot_op,
		none_op
	};
	
	typedef defaults <def_t> defaults_t;
}

#endif
