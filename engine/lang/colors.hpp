#ifndef COLORS_H_
#define COLORS_H_

// C/C++ headers
#include <ostream>

namespace zhetapi {

namespace detail {

struct Warning {};
struct Error {};

extern Warning warning;
extern Error error;

std::ostream &operator<<(std::ostream &, const Warning &);
std::ostream &operator<<(std::ostream &, const Error &);

}

}

#endif