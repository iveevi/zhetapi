#ifndef DEBUG_H
#define DEBUG_H

// C++ Standard Libraries
#include <iostream>
#include <functional>
#include <vector>
#include <sstream>

/* Functions to print the
 * C++ STLs for debugging:
 * 
 * fff */
template <class item>
std::string def_str(item i)
{
	std::ostringstream oss;
	oss << i;

	return oss.str();
}

template <class item>
void stl_reveal(std::ostream &os, const std::vector <item> &vec,
	std::function <std::string (item)> str = def_str <item>)
{
	os << "Vector Statistics: " << std::endl;
	os << "\tSize:" << vec.size() << std::endl;
	os << "Printing Vector:" << std::endl;

	for (item i : vec)
		os << "\t" << str(i) << std::endl;
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
