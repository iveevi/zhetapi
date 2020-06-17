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
	yydebug = 1;

	stree s = stree("x + 568 + 34.56 + 3/4 + 12i + (1 + 2i) + [3, 4, 5] + [[1, 0], [0, 1]]");

	cout << endl << "Stree:" << endl;

	s.print();
}
