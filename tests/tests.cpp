#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <complex>

#include <gmpxx.h>

#include "../engine/zcomplex.h"
#include "../engine/algorithms.h"
#include "../engine/calculus.h"
#include "../engine/combinatorial.h"
#include "../engine/expression.h"
#include "../engine/network.h"
#include "../engine/polynomial.h"
#include "../engine/rational.h"

using namespace std;

int main()
{
	polynomial <zcomplex <long double>> f {1, 4, 4, 0};

	vector <zcomplex <long double>> rts = f.roots(10000, 1E-500L, {0.4, 0.9});

	cout << endl << "rts:" << endl;

	for (auto val : rts)
		cout << "\t" << val << endl;

	auto ftrs = utility::solve_hlde_constant(f);

	cout << endl << "Solutions:" << endl;
	for (auto ftr : ftrs)
		cout << "\t" << ftr << endl;

	functor <double> ftr("f", {"x"}, "x^3 + x^2");

	cout << endl << "Function:" << endl << endl;

	ftr.print();

	cout << endl << "\t" << ftr << endl;
	cout << "\t" << ftr.differentiate("x") << endl;

	/* cout << endl << expression <double> ::in_place_evaluate("[2, 3].[5, 6]"); */

	stree s = stree("x + 568");

	cout << endl << "Stree:" << endl;

	s.print();
}
