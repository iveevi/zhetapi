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
#include <node_manager.hpp>
#include <barn.hpp>

namespace zhetapi {

#define BUF_SIZE	1024

	// Exception classes
	class type_mismatch {};
	class invalid_expr {};

	template <class O = int, class T = double, class U = int>
	O expr(std::string str, Barn <T, U> barn = Barn <T, U> ())
	{
		if (str.empty())
			throw invalid_expr();

		node_manager <T, U> out(str, barn);

		token *tptr = out.value();
		
		operand <O> *optr = dynamic_cast <operand <O> *> (tptr);

		if (optr == nullptr)
			throw type_mismatch();

		// in = optr->get();
		return optr->get();
	}
	
	template <class T = double, class U = int>
	std::string expr_str(std::string str, Barn <T, U> barn = Barn <T, U> ())
	{
		if (str.empty())
			throw invalid_expr();

		node_manager <T, U> out(str, barn);

		token *tptr = out.value();

		return tptr->str();
	}

	/* template <class O = int, class T = double, class U = int>
	void expressionf(O &in, const char *format, ...)
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
	void expressionf(O &in, Barn <T, U> brn, const char *format, ...)
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

	* template <class T = double, class U = int>
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
	} */

}

#endif
