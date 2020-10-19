// C/C++ headers
#include <ios>
#include <iostream>

// Engine headers
#include <function.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Function <double, int> f = "f(x, y) = x^2 + xyln(x) + y^2";
	Function <double, int> df = f.differentiate("x");

	cout << f << endl;
	cout << df << endl;

	node_manager <double, int> A("x^2 + x + y", {"x", "y"});
	node_manager <double, int> B("x^2 + x + y", {"x", "y"});

	cout << "\nA: " << A.display() << endl;
	cout << "B: " << B.display() << endl;

	cout << "\nMatches: " << std::boolalpha << node_manager <double, int> ::loose_match(A, B) << endl;
}
