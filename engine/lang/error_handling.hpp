#ifndef FEEDER_H_
#define FEEDER_H_

// C/C++ headers
#include <string>
#include <vector>

// Engine headers
#include <matrix.hpp>
#include <core/common.hpp>

namespace zhetapi {

// TODO: string matching...
size_t levenshtein(const std::string &, const std::string &);
Args suggestions(const std::string &, const Args &);

}

#endif
