#include <vector.hpp>

#include <std/linalg.hpp>

using namespace zhetapi;
using namespace std;

int main()
{
	Matrix <double> A {{1, 1}, {1, 0}};

	cout << linalg::qr_algorithm(A) << endl;

	Matrix <long double> B {{1, 1, 1}, {1, 1, 0}, {1, 0, 0}};

	cout << linalg::qr_algorithm(B) << endl;
	
	Matrix <long double> C = linalg::diag(1.0L, 2.0L, 3.0L, 4.0L);

	cout << linalg::qr_algorithm(C) << endl;
}
