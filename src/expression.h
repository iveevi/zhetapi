#ifndef EXPRESSION_H
#define EXPRESSION_H

// C++ Standard Libraries
#include <string>

// Custom Built Libraries
#include "operand.h"


// remove
using namespace tokens;

template <class T>
class expression {
	std::string m_cached;
public:
	expression(std::string = "");

	operand <T> evaluate(std::string = "") const;

	class empty_expr {};
	class invalid_expr {};
};

template <class T>
expression <T> ::expression(std::string str)
	: m_cached(str) {}

template <class T>
expression <T> ::evaluate(std::string str)
{
	if (str.empty())
		throw empty_expr();
	
	m_cached = str;
	
}

#endif