#ifndef CLASS_H_
#define CLASS_H_

// C/C++ headers
#include <string>

namespace zhetapi {

enum cls {
	c_none,
	c_polynomial,
	c_rational,
	c_exponential
};

extern std::string strclass[];

}

#endif
