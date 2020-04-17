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
#include "operation.h"
#include "var_stack.h"
#include "defaults.h"
#include "operand.h"
#include "stack.h"
#include "debug.h"

#include "node.h"

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

	// Includes caching, change to references
	const T &evaluate(std::string = "", var_stack <T> = var_stack <T> ());

	// Without caching
	static const T &in_place_evaluate(std::string = "", var_stack <T> = var_stack <T> ());

	// Without caching, and with formatting
	static const T&in_place_evaluate_formatted(const char *, ...);

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
const T &expression <T> ::evaluate(std::string str, var_stack <T> vst)
{
	if (str.empty() && m_cached.empty())
		throw invalid_expr();

	if (!str.empty())
		m_cached = str;

	//set_input(str.c_str());
	//e_scan_string(str.c_str());
	//operand <T> *out = new operand <T> ();
	//eparse(out);
	node *out = new node(str);
	return out->value();
}

template <class T>
const T &expression <T> ::in_place_evaluate(std::string str, var_stack <T> vst)
{
	if (str.empty())
		throw invalid_expr();

	// return value(symbols(str, vst));
	// set_input(str.c_str());
	//cout << "dumping contents of str:" << endl;
	//const char *s = str.c_str();
	//while (s)
	//	cout << *s << " @ " << (int)*s++ << endl;
	/* char *cpy = new char[str.length() + 1];
	int i;
	for (i = 0; i < str.length(); i++)
		cpy[i] = str[i];
	cpy[i] = '\n';

	* s = cpy;
	while (s)
		cout << *s << " @ " << (int)*s++ << endl; *
	e_scan_string(cpy);
	operand <T> *out = new operand <T> ();
	
	eparse(out, vst);
	return out->get(); */
	node *out = new node(str);
	return out->value();
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
	
	return in_place_evaluate(std::string(buf));
}

#endif
