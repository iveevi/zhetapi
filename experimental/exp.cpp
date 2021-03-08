#include <vector.hpp>

#include <std/linalg.hpp>

using namespace zhetapi;
using namespace std;

int main()
{
	using namespace zhetapi::linalg;

	Matrix <double> A {{1, 1, 1}, {1, 0, 1}, {2, 4, 5}};

	auto qr = qr_decompose(A);

	cout << "Q = " << qr.q() << endl;
	cout << "R = " << qr.r() << endl;

	auto lq = lq_decompose(A);
	
	cout << "L = " << lq.l() << endl;
	cout << "Q = " << lq.q() << endl;

	cout << "A = " << A << endl;

	A.swap_rows(0, 1);

	cout << "A = " << A << endl;

	Matrix <double> A_inv = A.inverse();

	cout << "A_inv = " << A_inv << endl;

	cout << "I3 = " << A * A_inv << endl;

	// Testing PSLQ
	Vector <long double> a = {3.14159265358, acos(-1)};

	auto rel1 = pslq(a);

	cout << string(50, '=') << endl;
	cout << "a = " << a << endl;
	cout << "rel1 = " << rel1 << endl;

	cout << "check = " << inner(a, rel1) << endl;

	long double phi = 1.61803398874989484820458683436;

	Vector <long double> b = {phi, phi * phi, 1};

	auto rel2 = pslq(b);

	cout << string(50, '=') << endl;
	cout << "b = " << b << endl;
	cout << "rel2 = " << rel2 << endl;

	cout << "check = " << inner(b, rel2) << endl;
}
