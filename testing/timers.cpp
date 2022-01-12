#include "global.hpp"

ostream &operator<<(ostream &os, const bench &b)
{
	tpoint tmp = clk.now();

	double mcs = chrono::duration_cast
		<chrono::microseconds>
		(tmp - b.epoch).count();
	
	string unit = " µs";

	if (mcs > 1e6) {
		mcs /= 1e6;
		unit = " s";
	} else if (mcs > 1e3) {
		mcs /= 1e3;
		unit = " ms";
	}

	os << byellow << "[";
	os << mcs;
	os << unit;
	os << "]\t" << reset;

	return os;
}
