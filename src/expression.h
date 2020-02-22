#ifndef EXPRESSION_H_
#define EXPRESSION_H_

// C++ Standard Libraries
#include <cstdio>
#include <cstdarg>
#include <string>

// Custom Built Libraries
#include "operand.h"
#include "parser.h"
#include "tree.h"
#include "stack.h"

// Future Note:
// Add parsing method
// for const char *
// parameters (overhead
// reduction from string
// class is possible)

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

	// Includes caching
	const T &evaluate(std::string = "", var_stack <T> = var_stack <T> ()) const;

	// Without caching
	static const T &in_place_evaluate(std::string = "", var_stack <T> = var_stack <T> ());

	// Without caching, and with formatting
	static const T&in_place_evaluate_formatted(const char *, ...);

	// User changeable buffer size
	static int BUF_SIZE;

	class invalid_expr {};
};

template <class T>
int expression <T> ::BUF_SIZE = 1000;

template <class T>
expression <T> ::expression(std::string str)
	: m_cached(str) {}

template <class T>
const T &expression <T> ::evaluate(std::string str, var_stack <T> vst) const
{
	if (str.empty() && m_cached.empty())
		throw invalid_expr();

	if (!str.empty())
		m_cached = str;

	tree <T> *eval = new tree <T> (m_cached, vst);
	token *temp = eval->value()->get();

	delete eval;
	return (dynamic_cast <operand <T> *> (temp))->get();
}

template <class T>
const T &expression <T> ::in_place_evaluate(std::string str, var_stack <T> vst)
{
	if (str.empty())
		throw invalid_expr();

	tree <T> eval(str, vst);
	return eval.value();
}

template <class T>
const T &expression <T> ::in_place_evaluate_formatted(const char *format, ...)
{
	va_list arg;
	int done;

	char buf[BUF_SIZE];
	
	va_start(arg, format);
	done = sprintf(buf, format, arg);
	va_end(arg);

	if (done < 0)
		throw invalid_expr();

	// Future Note:
	// remove overhead
	// with string class

	using namespace std;
	cout << "Buffer: " << buf << endl;
	
	return in_place_evaluate(std::string(buf));
}

#endif
