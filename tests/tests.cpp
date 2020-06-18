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
#include "../engine/scalar.h"

using namespace std;

int main()
{
	stree s = stree("x + 568 + 34.56 + 3/4 + 12i + (1 + 2i) + [3, 4, 5] + [[1, 0], [0, 1]]");

	cout << "Stree:" << endl;

	s.print();

	cout << scalar <int> () << endl;
	cout << scalar <double> (123.43) << endl;
	cout << scalar <rational <long int>> ({1, 2}) << endl;
	cout << scalar <zcomplex <long double>> ({1, 3}) << endl;
	cout << scalar <zcomplex <rational <long int>>> ({{1, 2}, {5, 6}}) << endl;
}