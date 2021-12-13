#include "global.hpp"

TEST(fourier_series)
{
	using namespace zhetapi;

	FourierSeries <double> fsa({1, 2, 3, 7}, {5, 6, 8});

	oss << "fsa @ 0 = " << fsa(0) << endl;
	oss << "fsa @ 10 = " << fsa(10) << endl;
	
	FourierSeries <double> fsb({1, 2, 5, 3, 6, 7, 8});
	
	oss << "fsb @ 0 = " << fsb(0) << endl;
	oss << "fsb @ 10 = " << fsb(10) << endl;

	if (fsa(0) != fsb(0) || fsa(10) != fsb(10)) {
		oss << "Unequal values..." << endl;

		return false;
	}

	return true;
}
