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
		data_t val;
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
		virtual data_t get() const;
		virtual const data_t &get() const;

		virtual data_t operator*() const;
		virtual const data_t &get() const;

		virtual type caller() override;
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
		gtree = new token_tree(exp);
		val = gtree->value()->dptr();
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
		gtree = new token_tree(exp);
		val = gtree->value()->dptr();
	}

	template <typename data_t>
	void group <data_t> ::operator[](std::string exp)
	{
		gtree = new token_tree(exp);
		val = gtree->value()->dptr();
	}

	template <typename data_t>
	data_t group <data_t> ::get()
	{
		return val;
	}

	template <typename data_t>
	const data_t &group <data_t> ::get()
	{
		return val;
	}

	template <typename data_t>
	data_t group <data_t> ::operator*()
	{
		return val;
	}

	template <typename data_t>
	const data_t &group <data_t> ::operator*()
	{
		return val;
	}
}

#endif