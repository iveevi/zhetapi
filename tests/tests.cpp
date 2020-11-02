// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <function.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Function <double, int> f = "f(x) = x^2 + xln(x) - x";

	cout << f << endl;

	f.print();

	Function <double, int> df = f.differentiate("x");

	cout << df << endl;

	df.print();
}
