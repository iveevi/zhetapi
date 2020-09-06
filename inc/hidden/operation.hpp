#ifndef OPERATION_H_
#define OPERATION_H_

// C++ Standard Libraries
#include <functional>
#include <string>
#include <vector>

// Engine Headers
#include <operand.hpp>

namespace zhetapi {

	/**
	 * @brief Represent a mathematical
	 * operation, which can base computations
	 * not only of operands, but also Variables,
	 * Functions, and other tokens. Type checking
	 * for appropriate template argument is left
	 * of the user, as mentioned later.
	 */
	class operation : public token {
	public:
		// Aliases
		using mapper = std::function <token *(const std::vector <token *> &)>;
	private:
		/* Member instance of the
		 * operation token class. */

		/**
		 * @brief The expected
		 * input format of the
		 * operation.
		 */
		std::string __input;

		/**
		 * @brief The expected
		 * output format of the
		 * operation; includes
		 * regex-like features.
		 */
		std::string __output;

		/**
		 * @brief The number of
		 * operands expected by
		 * the operation. Used to
		 * check input of computation.
		 */
		std::size_t ops;

		/**
		 * @brief The actual operation;
		 * takes a list of tokens and ouputs
		 * a single value. Type checking
		 * the arguments for appropriate
		 * template argument is left
		 * to the user or some higher level
		 * class such as config.
		 */
		mapper __opn;
	public:
		operation();
		operation(const operation &);
		operation(const std::string &, const std::string &,
				std::size_t,  mapper);

		token *operator()(const std::vector <token *> &) const;

		std::string fmt() const;
		std::string str() const override;

		std::size_t inputs() const;

		type caller() const override;

		token *copy() const override;

		bool operator==(token *) const override;

		class count_mismatch {};
		class token_mismatch {};
	};

	operation::operation() : __input(""), __output(""), ops(0) {}

	operation::operation(const operation &other)
	{
		__input = other.__input;
		__output = other.__output;
		ops = other.ops;
		__opn = other.__opn;
	}

	operation::operation(const std::string &in, const std::string &out, std::size_t
			opers, mapper fopn) : __input(in), __output(out), ops(opers),
			__opn(fopn) {}

	token *operation::operator()(const std::vector <token *> &ins) const
	{
		if (ins.size() != ops)
			throw count_mismatch();
		return __opn(ins);
	}

	std::string operation::fmt() const
	{
		return __input;
	}

	std::string operation::str() const
	{
		return "[" + __input + "](" + std::to_string(ops)
			+ ") - [" + __output + "]";
	}

	std::size_t operation::inputs() const
	{
		return ops;
	}

	token::type operation::caller() const
	{
		return opn;
	}

	token *operation::copy() const
	{
		return new operation(*this);
	}

	bool operation::operator==(token *t) const
	{
		operation *optr = dynamic_cast <operation *> (t);
		if (optr == nullptr)
			return false;

		return (ops == optr->ops)
			&& (__input == optr->__input)
			&& (__output == optr->__output);
	}

}

#endif
