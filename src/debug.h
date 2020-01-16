#ifndef DEBUG_H
#define DEBUG_H

// C++ Standard Libraries
#include <iostream>
#include <functional>
#include <vector>
#include <sstream>

// Custom Built Libraries
#include "token.h"

/* Functions to print the
 * C++ STLs for debugging:
 * 
 * fff */
std::string ptok(tokens::token *t)
{
	return t->str();
}

template <class item>
std::string def_str(item i)
{
	std::ostringstream oss;
	oss << i;

	return oss.str();
}

template <class item, class container>
void stl_reveal(std::ostream &os, const container &con,
	std::string (*str)(item))
{
	os << "Vector Statistics: " << std::endl;
	os << "\tSize:" << con.size() << std::endl;
	os << "Printing Vector:" << std::endl;

	for (item i : con)
		os << "\t" << str(i) << std::endl;
	os << std::endl;
}

/* Include alternate definitions
 * if debuggin mode is activated */
#ifdef DEBUG

#define dp_var(var) std::cout << "DEBUG | " << #var \
	<< ": " << var << std::endl;
#define dp_ptr(var) std::cout << "DEBUG | " << #var \
	<< ": " << *var << " @ " << var << std::endl;
#define dp_msg(msg) std::cout << "DEBUG | " << msg \
	<< std::endl;

/* Do nothing with the macros
 * if debug mode is inactive */
#else

#define dp_var(var)
#define dp_ptr(var)
#define dp_msg(msg)

#endif

#endif
