// C/C++ headers
#include <iostream>

// Engine headers
#include <function.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Function <double, int> f = "f(x) = x^2 + sin(x) + 4 * 6";

	cout << "f: " << f.display() << endl;

	f.print();
}