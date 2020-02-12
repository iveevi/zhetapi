#ifndef EXPRESSION_H
#define EXPRESSION_H

// C++ Standard Libraries
#include <string>

// Custom Built Libraries
#include "operand.h"
#include "tree.h"

template <class T>
class expression {
	// Change to a hash table
	std::string m_cached;
public:
	expression(std::string = "");

	// Includes caching
	const T &evaluate(std::string = "") const;

	class empty_expr {};
	class invalid_expr {};

	// Without caching
	static const T &in_place_evaluate(std::string = "");
};

template <class T>
expression <T> ::expression(std::string str)
	: m_cached(str) {}

template <class T>
const T &expression <T> ::evaluate(std::string str) const
{
	if (str.empty() && m_cached.empty())
		throw empty_expr();

	if (!str.empty())
		m_cached = str;

	tree <T> *eval = new tree <T> (m_cached);
	token *temp = eval->value()->get();

	delete eval;
	return (dynamic_cast <operand <T> *> (temp))->get();
}

template <class T>
const T &expression <T> ::in_place_evaluate(std::string str)
{
	if (str.empty())
		throw empty_expr();

	tree <T> *eval = new tree <T> (str);
	token *temp = eval->value()->get();

	delete eval;
	return (dynamic_cast <operand <T> *> (temp))->get();
}

#endif
