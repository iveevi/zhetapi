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
#include <engine.hpp>

#include <core/node_manager.hpp>

namespace zhetapi {

#define BUF_SIZE 1024

// Exception classes
class type_mismatch {};
class invalid_expr {};

template <class O = int, class T = double, class U = int>
O expr(std::string str, Engine <T, U> engine = Engine <T, U> ())
{
	if (str.empty())
		throw invalid_expr();

	node_manager <T, U> out(str, engine);

	Token *tptr = out.value();
	
	Operand <O> *optr = dynamic_cast <Operand <O> *> (tptr);

	if (optr == nullptr)
		throw type_mismatch();

	// in = optr->get();
	return optr->get();
}

template <class T = double, class U = int>
::std::string expr_str(::std::string str, Engine <T, U> engine = Engine <T, U> ())
{
	if (str.empty())
		throw invalid_expr();

	node_manager <T, U> out(str, engine);

	Token *tptr = out.value();

	return tptr->str();
}

}

#endif
