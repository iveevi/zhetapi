// C/C++ headers
#include <ios>
#include <iostream>

// Engine headers
#include <function.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Function <double, int> f = "f(x, y) = x^2 + xyln(x) - y^(2 + 5) + 4 * 5";

	cout << f << endl;

	// Function <double, int> df = f.differentiate("x");

	// cout << df << endl;
}
