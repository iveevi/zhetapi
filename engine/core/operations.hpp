#ifndef OPERATIONS_H_
#define OPERATIONS_H_

// Standard headers
#include <cstdint>

namespace zhetapi {

namespace core {

// 64 bits is sufficient
using OID = uint64_t;

// Operation codes
enum OpCode : OID {
	l_get,
	l_const,
	l_add,
	l_sub,
	l_mul,
	l_div
};

// Quick id converter
constexpr OID ovlid(OID a, OID b)
{
	return a + (b << 32);
}

}

}

#endif