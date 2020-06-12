#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <gmpxx.h>

#include "expression.h"
#include "algorithms.h"
#include "combinatorial.h"
#include "network.h"

using namespace std;

int main()
{
	vector <pair <double, double>> D {
		{0, 0},
		{1, 3},
		{2, 4},
		{5, 5},
		{9, 14}
	};

	table <double> tbl;

	functor <double> ftr("f(a, b, c, x) = ax^2 + bx + c");
	tbl.insert_ftr(ftr);

	functor <double> ftr_dx = ftr.differentiate("x");
	ftr_dx.rename("g");
	tbl.insert_ftr(ftr_dx);

	functor <double> c("C(a, b, c, x, y) = (f(a, b, c, x) - y)^2", tbl);

	functor <double> c_dx("Cp(a, b, c, x, y) = (g(a, b, c, x) - y)^2", tbl);

	cout << ftr << endl;
	cout << ftr_dx << endl;
	cout << c << endl;
	cout << c_dx << endl;

	cout << "2 + 5 - 7 * 7: " << expression <double> ::in_place_evaluate("2 + 5 - 7 * 7") << endl;

	cout << utility::binom(476.0, 2.0) << endl;
}
