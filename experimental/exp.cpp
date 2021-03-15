#include <complex.hpp>
#include <polynomial.hpp>

using namespace zhetapi;
using namespace std;

int main()
{
	// Test-global resources
	double coeffs[] {1, 2, 3, 4};

	// Tests
	Polynomial <double> f {1, 2, 3, 4, 5};
	
	cout << "f: " << f << endl;
	cout << "\tdeg(f) = " << f.degree() << endl;
	cout << "\tf(1) = " << f(1) << endl;
	cout << "\tf(1) = " << f.evaluate(1) << endl;
	
	Polynomial <double> g {1, 2, 3, 4};

	cout << "g: " << g << endl;
	cout << "\tdeg(f) = " << g.degree() << endl;
	cout << "\tg(1) = " << g(1) << endl;
	cout << "\tg(1) = " << g.evaluate(1) << endl;

	Polynomial <double> h(coeffs, 3);
	
	cout << "h: " << h << endl;
	cout << "\tdeg(h) = " << h.degree() << endl;
	cout << "\th(1) = " << h(1) << endl;
	cout << "\th(1) = " << h.evaluate(1) << endl;
	
	Polynomial <double> fp1 {1, 2, 3, 4, 5};
	Polynomial <double> fp2 {1, 2, 3, 4, 6};
	Polynomial <double> fp3 {1, 2, 3, 4};

	cout << boolalpha;
	cout << "f == fp1: " << (f == fp1) << endl;
	cout << "f == fp2: " << (f == fp2) << endl;
	cout << "f == fp3: " << (f == fp3) << endl;

	Polynomial <double> fcpy1(f);
	Polynomial <double> fcpy2 = f;

	cout << "fcpy1 = " << fcpy1 << endl;
	cout << "fcpy2 = " << fcpy2 << endl;

	cout << "f == fcpy1: " << (f == fcpy1) << endl;
	cout << "f == fcpy2: " << (f == fcpy2) << endl;

	cout << "f + g = " << f + g << endl;
	cout << "f - g = " << f - g << endl;
	cout << "g - f = " << g - f << endl;
}
