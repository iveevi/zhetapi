#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <complex>

#include <gmpxx.h>

#include "zcomplex.h"
#include "algorithms.h"
#include "calculus.h"
#include "combinatorial.h"
#include "expression.h"
#include "network.h"
#include "polynomial.h"
#include "rational.h"

using namespace std;

bool operator<(const zcomplex <long double> &a, const zcomplex <long double> &b)
{
	return norm(a) < norm(b);
}

bool operator<=(const zcomplex <long double> &a, const zcomplex <long double> &b)
{
	return norm(a) <= norm(b);
}

int main()
{
	polynomial <zcomplex <long double>> f {1, 4, 4, 0};

	vector <zcomplex <long double>> rts = f.roots(10000, 1E-500L, {0.4, 0.9});

	cout << endl << "rts:" << endl;

	for (auto val : rts)
		cout << "\t" << val << endl;

	auto ftrs = utility::solve_hlde_constant(f);

	cout << "Solutions:" << endl;
	for (auto ftr : ftrs)
		cout << "\t" << ftr << endl;
}
