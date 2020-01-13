#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>

#ifdef DEBUG

#define dp_var(var) std::cout << "DEBUG | " << #var \
	<< ": " << var << std::endl;
#define dp_ptr(var) std::cout << "DEBUG | " << #var \
	<< ": " << *var << " @ " << var << std::endl;
#define dp_msg(msg) std::cout << "DEBUG | " << msg \
	<< std::endl;

#else

#define dp_var(var)
#define dp_ptr(var)
#define dp_msg(msg)

#endif

#endif
