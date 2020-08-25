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
	string line;

	while (getline(cin, line)) {
		cout << "In: " << line << endl;

		// node <double, int> nd = string(line);

		// cout << "Out: " << (expr <double, int> (line))->str() << "\n" << endl;
		// cout << "Out: " << (expr(line))->str() << "\n" << endl;
		// cout << "Out: " << (expr(line))->str() << "\n" << endl;
		//cout << "Out: " << (cmp.expr(line))->str() << "\n" << endl;
		cout << "Out: " << expr(line) << "\n" << endl;
	}

	Function <double, int> f("f(x) = x^2");

	cout << f << endl;


	f.print();

	cout << (f(10.0))->str() << endl;
	
	Function <double, int> g("g(x) = x . [1, 2, 3]");
	
	cout << (g(Vector <int> {1, 2, 3}))->str() << endl;

	Function <double, int> dg = g.differentiate(0);

	cout << endl << dg << endl;

	dg.print();
}
