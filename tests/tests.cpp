#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <complex>
#include <type_traits>

#include <gmpxx.h>

/*
#include "../engine/barn.h"
#include "../engine/node.h"
#include "../engine/polynomial.h"
#include "../engine/rational.h"
#include "../engine/zcomplex.h"
*/

#include <barn.h>
#include <node.h>
#include <polynomial.h>
#include <rational.h>
#include <zcomplex.h>

using namespace std;

int main()
{
	string line;

	while (getline(cin, line)) {

		node <double, int> nd = string(line);

		cout << endl << "POST COMPRESSION/EVALUATION:" << endl;
		nd.print();
	}
}
