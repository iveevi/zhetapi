// C/C++ headers
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <complex>
#include <type_traits>

// Numerical headers
#include <gmpxx.h>

// Engine headers
#include <barn.h>
#include <complex.h>
#include <function.h>
#include <node.h>
#include <polynomial.h>
#include <rational.h>
#include <expression.h>

using namespace std;

int main()
{
	string line = "3 + 4";

	cout << line << ": " << expr(line) << endl;

	cout << "sizeof(node): " << sizeof(node <double, int>) << endl;
	cout << "sizeof(barn): " << sizeof(Barn <double, int>) << endl;
	cout << "sizeof(barn *): " << sizeof(Barn <double, int> *) << endl;
}
