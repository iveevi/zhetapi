#ifndef COMMON_H_
#define COMMON_H_

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

#endif
