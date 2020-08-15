#include <iostream>

#include "../engine/matrix.h"
#include "../engine/polynomial.h"
#include "../engine/rational.h"

using namespace std;

int main()
{
	matrix <rational <int>> A {
		{1, 2, 4, 4},
		{1, 16, 256, 4}
	};

	/* matrix <rational <int>> A {
		{1, 2, 4, 8, 4},
		{1, 9, 81, 729, 9},
		{1, 16, 256, 4096, 4}
	}; */

	cout << "A:" << endl << A << endl;

	for (size_t i = 0; i < A.get_rows() - 1; i++) {
		if (A[i][i] != 1) 
		cout << "'pivot': " << A[i][i] << endl;
	}
}
