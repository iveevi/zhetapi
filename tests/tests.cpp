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
	Function <double, int> f, df;

	cout << "======================================" << endl;

        f = "f(x) = 43x + x^2 + sin(x)/x + cos(x^2)";

	cout << endl << f << endl;
	f.print();

	df = f.differentiate("x");

	cout << endl << df << endl;
	df.print();

	cout << "======================================" << endl;

        f = "f(x) = lg(x) + log(2, x)";

	cout << endl << f << endl;
	f.print();

	df = f.differentiate("x");

	cout << endl << df << endl;
	df.print();
}
