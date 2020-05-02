#ifndef DEBUG_H_
#define DEBUG_H_

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
template <class T>
std::string to_str(T i)
{
	std::ostringstream oss;
	oss << i;

	return oss.str();
}

template <class T, class V, class P = std::string (*)(T)>
void stl_reveal(T dm, const V &con,
	P str = to_str, std::ostream
	&os = std::cout)
{
	os << "Vector Statistics: " << std::endl;
	os << "\tSize:" << con.size() << std::endl;
	os << "Printing Vector:" << std::endl;
	
	for (T i : con)
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
