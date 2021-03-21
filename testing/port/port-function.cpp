#include "port.hpp"

bool function_computation(ostringstream &oss)
{
	using namespace zhetapi;

	bench tb;

	const int iters = 1;

	Function f = "f(x, y) = x^2 + 2x + ln(x^3 * y - 36)";

	tb = bench();
	for (int i = 0; i < iters; i++)
		f(5, 4);
	
	oss << "Time for regular computation: " << tb << endl;

	f.set_threads(8);

	tb = bench();
	for (int i = 0; i < iters; i++)
		f(5, 4);
	
	oss << "Time for multi-threaded computation: " << tb << endl;

	return true;
}