// C/C++ headers
#include <iostream>

// Engine headers
#include <function.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Function <double, int> fx = "f(x) = x^2 + x * ln(x)";

	cout << fx(10)->str() << endl;

	typedef zhetapi::token *(*ftr)(zhetapi::token *);

	ftr gfx = (ftr) fx.compile_general();

	zhetapi::token *opd = new operand <int> (10);

	cout << gfx(opd)->str() << endl;

	delete opd;
}
