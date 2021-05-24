#ifndef COMMON_H_
#define COMMON_H_

// C/C++ heaaders
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Engine headers
#include "../token.hpp"

namespace zhetapi {

// Type aliases
using Args = std::vector <std::string>;
using Targs = std::vector <Token *>;

using Pardon = std::set <std::string>;

template <class T>
using Symtab = std::unordered_map <std::string, T>;

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

}

#endif
