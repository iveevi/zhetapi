#include "port.hpp"

bench mark;

ostream &operator<<(ostream &os, const bench &b)
{
	tmp = clk.now();

	os << "[";
	os << chrono::duration_cast<chrono::microseconds>(tmp - epoch).count();
	os << "]\t";

	return os;
}