#ifndef FEEDER_H_
#define FEEDER_H_

// C/C++ headers
#include <string>

// Engine headers
#include <martix.hpp>

namespace zhetapi {

// TODO: string matching...
size_t levenshtein(const std::string &a, const std::string &b)
{
	size_t rows = a.length() + 1;
	size_t cols = b.length() + 1;

	// Make the matrix
	Matrix <size_t> (rows, cols, 0);
}

}

#endif
