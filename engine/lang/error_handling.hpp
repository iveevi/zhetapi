#ifndef ERROR_HANDLING_H_
#define ERROR_HANDLING_H_

// C/C++ headers
#include <string>
#include <vector>

// Engine headers
#include "../matrix.hpp"
#include "../core/common.hpp"

namespace zhetapi {

// Forward declarations
class Engine;

// TODO: add other string matching metrics (levenshtein is not very natural)
size_t levenshtein(const std::string &, const std::string &);
Args symbol_suggestions(const std::string &, const Args &);
void symbol_error_msg(const std::string &, const std::string &, Engine *);

}

#endif
