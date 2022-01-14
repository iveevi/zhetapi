#ifndef PRINT_H_
#define PRINT_H_

// Standard headers
#include <string>
#include <vector>

namespace zhetapi {

namespace io {

using Args = std::vector <std::string>;

std::string table(const Args &, const std::vector <Args> &);

}

}

#endif
