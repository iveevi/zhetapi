#include "../include/polynomial.hpp"

namespace zhetapi {

// Next power of 2
size_t npow2(size_t k)
{
	size_t v = k;

	// Assuming size_t is 64-bit
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;

	return v;
}

}
