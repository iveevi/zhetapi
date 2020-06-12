#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <gmpxx.h>

#include "algorithms.h"
#include "combinatorial.h"
#include "expression.h"
#include "network.h"
#include "rational.h"

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

	cout << endl << "2 + 5 - 7 * 7: " << expression <double> ::in_place_evaluate("2 + 5 - 7 * 7") << endl;

	cout << endl << "C(10, 2) = " << utility::binom(10.0, 2.0) << endl;
	cout << "Falling(5, 5) = " << utility::falling_power(5, 5) << endl;
	cout << "Factorial(6) = " << utility::integral_factorial(6) << endl;
	cout << "C(10, 2) = " << utility::integral_binom(10, 2) << endl;
	cout << "gcd(10, 6) = " << utility::gcd(10.0, 6.0) << endl;
	cout << "gcd(10, 6) = " << utility::integral_gcd(100, 175) << endl;

	vector <double> brs = utility::bernoulli_sequence_real(12.0);

	cout << endl << "First 13 Bernoulli Numbers:" << endl;

	size_t counter = 0;
	for (auto pr : brs)
		cout << "\t" << counter++ << ":\t" << pr << endl;

	rational <int> a(45, 25);
	rational <int> b(6, 11);

	cout << endl << "a:\t" << a << "\t" << (double) a << endl;
	cout << "b:\t" << b << "\t" << (double) b << endl;

	a += b;
	cout << endl << "a: " << a << endl;

	a -= b;
	cout << "a: " << a << endl;

	a *= b;
	cout << "a: " << a << endl;

	a /= b;
	cout << "a: " << a << endl;
}
