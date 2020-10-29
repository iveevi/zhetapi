// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <api.hpp>
#include <operand.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	/* Function <double, int> f = "f(x) = x^2 + xln(x) - x";

	cout << f << endl;

	f.print();

	Function <double, int> df = f.differentiate("x");

	cout << df << endl;

	df.print(); */

	vector <Token *> tmp {
		new Operand <int> (35),
		new Operand <double> (35),
		new Operand <string> ("fdfs"),
	};

	auto tpl = zhetapi_cast <Operand <int> *, Operand <double> *, Operand <string> *> (tmp);
}
