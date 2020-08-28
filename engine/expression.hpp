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
#include <node.hpp>
#include <barn.hpp>
#include <vtable.hpp>

#define BUF_SIZE	1024

// Exception class
class type_mismatch {};
class invalid_expr {};

template <class O = int, class T = double, class U = int>
void expr(O &in, std::string str, vtable <T> tbl = vtable <T> ())
{
	if (str.empty())
		throw invalid_expr();

	node <T, U> out(str, tbl);

	token *tptr = out.value();
	
	operand <O> *optr = dynamic_cast <operand <O> *> (tptr);

	if (optr == nullptr)
		throw type_mismatch();

	in = optr->get();
}

template <class O = int, class T = double, class U = int>
void exprf(O &in, const char *format, ...)
{
	va_list arg;
	int done;

	char buf[BUF_SIZE];
	
	va_start(arg, format);
	done = sprintf(buf, format, arg);
	va_end(arg);

	if (done < 0)
		throw invalid_expr();
	
	expr <O, T, U> (in, std::string(buf));
}

template <class O = int, class T = double, class U = int>
void exprf(O &in, Barn <T, U> brn, const char *format, ...)
{
	va_list arg;
	int done;

	char buf[BUF_SIZE];
	
	va_start(arg, format);
	done = sprintf(buf, format, arg);
	va_end(arg);

	if (done < 0)
		throw invalid_expr();
	
	expr <O, T, U> (in, std::string(buf));
}

template <class T = double, class U = int>
std::string expr(std::string str, vtable <T> tbl = vtable <T> ())
{
	if (str.empty())
		throw invalid_expr();

	node <T, U> out(str, tbl);

	token *tptr = out.value();

	return tptr->str();
}

template <class T = double, class U = int>
std::string exprf(const char *format, ...)
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

template <class T = double, class U = int>
std::string exprf(Barn <T, U> brn, const char *format, ...)
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
