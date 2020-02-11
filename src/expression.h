#ifndef EXPRESSION_H
#define EXPRESSION_H

// C++ Standard Libraries
#include <string>

// Custom Built Libraries
#include "operand.h"
#include "token_tree.h"


// remove
using namespace tokens;

template <class T>
class expression {
	std::string m_cached;
public:
	expression(std::string = "");

	operand <T> *evaluate(std::string = "") const;

	class empty_expr {};
	class invalid_expr {};
};

template <class T>
expression <T> ::expression(std::string str)
	: m_cached(str) {}

template <class T>
operand <T> *expression <T> ::evaluate(std::string str)
{
	if (str.empty() && m_cached.empty())
		throw empty_expr();

	if (!str.empty())
		m_cached = str;

	token_tree <T> *eval = new token_tree <T> (m_cached);
	token *temp = eval->value()->get();

	delete eval;
	return dynamic_cast <operand <T> *> (temp);
}

#endif
