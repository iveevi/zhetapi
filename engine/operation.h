#ifndef OPERATION_H_
#define OPERATION_H_

// C++ Standard Libraries
#include <functional>
#include <string>
#include <vector>

// Engine Headers
#include "operand.h"

/**
 * @brief Represent a mathematical
 * operation, which can base computations
 * not only of operands, but also variables,
 * functors, and other tokens. Type checking
 * for appropriate template argument is left
 * of the user, as mentioned later.
 */
template <class T>
class operation : public token {
	/* Member instance of the
	 * operation token class. */

	/**
	 * @brief The expected
	 * input format of the
	 * operation.
	 */
	std::string input;

	/**
	 * @brief The expected
	 * output format of the
	 * operation; includes
	 * regex-like features.
	 */
	std::string output;

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
	std::function <token *(const std::vector <token *> &)> opn;
public:
	operation();
	operation(const operation &);
	operation(const std::string &, const std::string &,
			std::size_t,  std::function
			<token *(const std::vector <token *> &)>);

	token *operator()(const std::vector <token *> &) const;

	std::string fmt() const;
	std::string str() const override;

	type caller() const override;

	token *copy() const override;

	bool operator==(token *) const override;

	class count_mismatch {};
	class token_mismatch {};
};

template <class T>
operation <T> ::operation() : input(""), output(""), ops(0) {}

template <class T>
operation <T> ::operation(const operation &other)
{
	input = other.input;
	output = other.output;
	ops = other.ops;
	opn = other.opn;
}

template <class T>
operation <T> ::operation(const std::string &in, const std::string &out,
		std::size_t opers, std::function
		<token *(const std::vector <token *> &)> fopn) : input(in),
		output(out), ops(opers), opn(fopn) {}

template <class T>
token *operation <T> ::operator()(const std::vector <token *> &ins) const
{
	if (ins.size() != ops)
		throw count_mismatch();
	return opn(ins);
}

template <class T>
std::string operation <T> ::fmt() const
{
	return input;
}

template <class T>
std::string operation <T> ::str() const
{
	return "[" + input + "](" + std::to_string(ops)
		+ ") - [" + output + "]";
}

template <class T>
token::type operation <T> ::caller() const
{
	return OPERATION;
}

template <class T>
token *operation <T> ::copy() const
{
	return new operation(*this);
}

template <class T>
bool operation <T> ::operator==(token *t) const
{
	if (t->caller() != token::OPERATION)
		return false;

	return (ops == (dynamic_cast <operation *> (t))->ops)
		&& (input == (dynamic_cast <operation *> (t))->input)
		&& (output == (dynamic_cast <operation *> (t))->output);
}

#endif
