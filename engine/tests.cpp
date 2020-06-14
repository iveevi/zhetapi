#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <gmpxx.h>

#include "algorithms.h"
#include "expression.h"
#include "network.h"
#include "combinatorial.h"
#include "rational.h"
#include "polynomial.h"

using namespace std;

int main()
{
	functor <double> f("f(x) = sum^{x}_{i = 0} i");
	functor <double> g("g", {"x"}, "sum_{i = 0}^{x} i");

	functor <double> p("p", {"i"}, "i^2 + i");

	cout << f << endl;
	cout << g << endl;

	cout << endl << "f @ 10:\t" << f(10) << endl;
	cout << "g @ 10:\t" << g(10) << endl;

	cout << endl << "Pre Classification:";
	p.print();

	if (p.classify() == c_polynomial)
		cout << endl << "Function p is polynomic" << endl;

	cout << endl << "Post Classification:" << endl;
	p.print();

	polynomial <double> a({1, 2, 4, 0});

	cout << endl << a << endl;
	cout << "a(1): " << a(2) << endl;

	polynomial <double> b {1, 2, 0, 5, 0, 4};

	cout << endl << b << endl;
	cout << "b(1): " << b(2) << endl;

	cout << endl << b.differentiate() << endl;

	pair <polynomial <double>, double> pr = a.synthetic_divide(2);

	cout << endl << "q: " << pr.first << endl;
	cout << "r: " << pr.second << endl;
}
