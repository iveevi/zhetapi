#include <vector.hpp>

#include <std/linalg.hpp>

using namespace zhetapi;
using namespace std;

int main()
{
	Vector <double> a {1, 2, 3};
	Vector <double> b {5, 4, 3};
	Vector <double> c {7, 6, 3};

	cout << Matrix <double> {a, b, c} << endl;

	cout << linalg::proj(a, b) << endl;

	Matrix <double> A {a, b, c};

	auto qr = linalg::qr_decompose(A);

	cout << qr.first << endl;
	cout << qr.second << endl;
}