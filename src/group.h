#ifndef GROUP_H
#define GROUP_H

// Custom Built Libraries
#include "token.h"
#include "token_tree.h"

namespace tokens {
	/* Beginning of grouping token */
        template <typename data_t>
        class group : public token {
		/* stored tree to represent
		 * the expression inside
		 * parenthesis */
		trees::token_tree <data_t> *gtree;

		// Store the value of tree to
		// retain quick calling
		operand <data_t> *val;
	public:
		/* string constructor
		(inside parenthesis) */
		group(std::string);

		/* Default construtor */
		group();

		// Change input
		virtual void set(std::string);
		virtual void operator[](std::string);

		// Get value
		virtual operand <data_t> *get() const;
		virtual operand <data_t> *operator*() const;

		virtual type caller() const override;
		virtual std::string str() const override;
        };

	/* Group Class Member Functions
	 *
	 * See class declaration for a
	 * description of each function
	 *
	 * Constructors: */
	template <typename data_t>
	group <data_t> ::group(std::string exp)
	{
		gtree = new trees::token_tree <data_t> (exp);
		val = gtree->value()->dptr()->get_opn_t();
	}

	template <typename data_t>
	group <data_t> ::group()
	{
		gtree = nullptr;
		val = data_t();
	}

	/* Virtualized member functions:
	 * setters, getter and operators */
	template <typename data_t>
	void group <data_t> ::set(std::string exp)
	{
		gtree = new trees::token_tree <data_t> (exp);
		val = gtree->value()->dptr()->get_opn_t();
	}

	template <typename data_t>
	void group <data_t> ::operator[](std::string exp)
	{
		gtree = new trees::token_tree <data_t> (exp);
		val = gtree->value()->dptr()->get_opn_t();
	}

	template <typename data_t>
	operand <data_t> *group <data_t> ::get() const
	{
		return val;
	}

	template <typename data_t>
	operand <data_t> *group <data_t> ::operator*() const
	{
		return val;
	}

	// functions inherited
	template <typename data_t>
	token::type group <data_t> ::caller() const
	{
		return GROUP;
	}

	template <typename data_t>
	std::string group <data_t> ::str() const
	{
		// Overload the token_tree
		// ostream operator and add
		// a string method to it
		return "Add later";
	}
}

#endif