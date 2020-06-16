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
	functor <double> ftr {"f", {"x", "y"}, "2 + 4x + 6 + 6y^2 + 7y^2 + 1"};

	cout << string(30, '_') << endl;
	cout << ftr << endl << string(30, '=') << endl;
	ftr.print();
	
	node <double> a {"2 + 2xy", table <double> (), {{"x", true}, {"y", true}}};
	node <double> b {"2yx + 2", table <double> (), {{"x", true}, {"y", true}}};

	cout << endl << "Nodes:" << endl;
	a.print();
	b.print();

	bool bl = a.matches(b);

	cout << endl << std::boolalpha << "Matching: " << bl << endl;

	/* network <double> c (vector <size_t> {3, 5, 6, 1});

	cout << "Networks:" << endl;
	c.print(); */
}
