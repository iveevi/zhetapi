#ifndef COMMON_H_
#define COMMON_H_

// C/C++ heaaders
#include <vector>
#include <set>
#include <string>

// Type aliases
using Args = std::vector <std::string>;
using Pardon = std::set <std::string>;

// This file contains miscellaneous functions that
// do not really fit into any other header
template <class T>
void collect(std::vector <T> &bin, T x)
{
	bin.push_back(x);
}

template <class T, class ... U>
void collect(std::vector <T> &bin, T x, U ... rest)
{
	bin.push_back(x);

	collect(bin, rest...);
}

Args args_union(const Args &, const Args &);

#endif
