#include "port.hpp"

ostream &operator<<(ostream &os, const bench &b)
{
	tpoint tmp = clk.now();

	os << "[";
	os << chrono::duration_cast<chrono::microseconds>(tmp - b.epoch).count();
	os << " µs]\t";

	return os;
}