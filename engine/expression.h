#ifndef EXPRESSION_H_
#define EXPRESSION_H_

// C++ Standard Libraries
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <stack>
#include <queue>

// Custom Built Libraries
#include "node.h"

/* FUTURE: Remove anything related
 * to caching in this (expression)
 * class.
 *
 * REASON: Caching, if it is to be
 * implemented, will be done at a
 * higher level, such as assignment
 * or application, rather than
 * expression.
 *
 * CONSEQUENCE: The expression class
 * should be left with only static methods,
 * and these methods should be placed outside
 * this class; destory the expression class,
 * place them in a namespace if necessary. */

template <class T>
class expression {
	// Future Note:
	// change string operations
	// into const char *
	// operations, also
	// change to a hash table
	std::string m_cached;

	// Future Note:
	// Add later:
	// splay_stack <std::string> m_cached;
public:
	expression(std::string = "");

	// Includes caching, change to references
	T evaluate(std::string = "", table <T> = table <T> ());

	// Without caching
	static T in_place_evaluate(std::string = "", table <T> = table <T> ());

	// Without caching, and with formatting
	static T in_place_evaluate_formatted(const char *, ...);

	// User changeable buffer size
	static int BUF_SIZE;

	class invalid_expr {};
	class syntax_error {};
};

template <class T>
int expression <T> ::BUF_SIZE = 1000;

template <class T>
expression <T> ::expression(std::string str)
	: m_cached(str) {}

template <class T>
T expression <T> ::evaluate(std::string str, table <T> tbl)
{
	if (str.empty() && m_cached.empty())
		throw invalid_expr();

	if (!str.empty())
		m_cached = str;

	node <T> *out = new node <T> (str, tbl);

	T val = out->value();
	delete out;

	return val;
}

template <class T>
T expression <T> ::in_place_evaluate(std::string str, table <T> tbl)
{
	if (str.empty())
		throw invalid_expr();

	node <T> *out = new node <T> (str, tbl);

	T val = out->value();
	delete out;

	return val;
}

template <class T>
T expression <T> ::in_place_evaluate_formatted(const char *format, ...)
{
	va_list arg;
	int done;

	char buf[BUF_SIZE];
	
	va_start(arg, format);
	done = sprintf(buf, format, arg);
	va_end(arg);

	if (done < 0)
		throw invalid_expr();
	
	return in_place_evaluate(std::string(buf));
}

#endif
