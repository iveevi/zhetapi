#ifndef EXPRESSION_H_
#define EXPRESSION_H_

// C/C++ headers
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <iterator>
#include <queue>
#include <stack>
#include <string>

// Engine headers
#include <node.h>
#include <barn.h>
#include <vtable.h>

#define BUF_SIZE	1024

// Exception class
class invalid_expr {};

template <class T, class U>
token *expr(std::string str, vtable <T> tbl = vtable <T> ())
{
	if (str.empty())
		throw invalid_expr();

	node <T, U> *out = new node <T, U> (str, tbl);

	return out->get_token();
}

template <class T, class U>
token *exprf(const char *format, ...)
{
	va_list arg;
	int done;

	char buf[BUF_SIZE];
	
	va_start(arg, format);
	done = sprintf(buf, format, arg);
	va_end(arg);

	if (done < 0)
		throw invalid_expr();
	
	return expr <T, U> (std::string(buf));
}

template <class T, class U>
token *exprf(barn <T, U> brn, const char *format, ...)
{
	va_list arg;
	int done;

	char buf[BUF_SIZE];
	
	va_start(arg, format);
	done = sprintf(buf, format, arg);
	va_end(arg);

	if (done < 0)
		throw invalid_expr();
	
	return expr <T, U> (std::string(buf));
}

#endif
