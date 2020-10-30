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
		new Operand <double> (432423),
		new Operand <string> ("fdfs"),
	};

	Operand <int> *tptr1;
	Operand <double> *tptr2;
	Operand <string> *tptr3;
	Operand <long int> *fake;

	zhetapi_cast(tmp, tptr1, tptr2, tptr3, fake);

	cout << tptr1->str() << endl;
	cout << tptr2->str() << endl;
	cout << tptr3->str() << endl;
}
