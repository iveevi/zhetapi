#include "port.hpp"

TEST(polynomial_construction)
{
	using namespace zhetapi;

	// Test-global resources
	int coeffs[] {1, 2, 3, 4};

	// Tests
	Polynomial <int> f {1, 2, 3, 4, 5};
	
	oss << "f: " << f << endl;
	oss << "\tdeg(f) = " << f.degree() << endl;
	oss << "\tf(1) = " << f(1) << endl;
	oss << "\tf(1) = " << f.evaluate(1) << endl;
	
	if (f.degree() != 5) {
		oss << "INCORRECT DEGREE (for f)" << endl;

		return false;
	}

	if (f(1) != 15) {
		oss << "INCORRECT VALUE (for f)" << endl;

		return false;
	}
	
	Polynomial <int> g {1, 2, 3, 4};

	oss << "g: " << g << endl;
	oss << "\tdeg(f) = " << g.degree() << endl;
	oss << "\tg(1) = " << g(1) << endl;
	oss << "\tg(1) = " << g.evaluate(1) << endl;

	if (g.degree() != 3) {
		oss << "INCORRECT DEGREE (for g)" << endl;

		return false;
	}
	
	if (g(1) != 10) {
		oss << "INCORRECT VALUE (for g)" << endl;

		return false;
	}

	Polynomial <int> h(coeffs, 3);
	
	oss << "h: " << h << endl;
	oss << "\tdeg(h) = " << h.degree() << endl;
	oss << "\th(1) = " << h(1) << endl;
	oss << "\th(1) = " << h.evaluate(1) << endl;
	
	if (h.degree() != 2) {
		oss << "INCORRECT DEGREE (for h)" << endl;

		return false;
	}
	
	if (h(1) != 6) {
		oss << "INCORRECT VALUE (for h)" << endl;

		return false;
	}

	return true;
}

TEST(polynomial_comparison)
{
	using namespace zhetapi;

	Polynomial <double> f {1, 2, 3, 4, 5};
	Polynomial <double> fp1 {1, 2, 3, 4, 5};
	Polynomial <double> fp2 {1, 2, 3, 4, 6};
	Polynomial <double> fp3 {1, 2, 3, 4};

	Polynomial <double> fcpy1(f);
	Polynomial <double> fcpy2 = f;
	
	oss << boolalpha;
	oss << "f == fp1: " << (f == fp1) << endl;
	oss << "f == fp2: " << (f == fp2) << endl;
	oss << "f == fp3: " << (f == fp3) << endl;

	// TODO: Add assert tests
	if (f != fp1 || f == fp2 || f == fp3)
		return false;

	oss << "fcpy1 = " << fcpy1 << endl;
	oss << "fcpy2 = " << fcpy2 << endl;

	oss << "f == fcpy1: " << (f == fcpy1) << endl;
	oss << "f == fcpy2: " << (f == fcpy2) << endl;
	
	if (f != fcpy1 || f != fcpy2)
		return false;

	return true;
}

TEST(polynomial_arithmetic)
{
	using namespace zhetapi;

	Polynomial <int> f {1, 2, 3, 4, 5};
	Polynomial <int> g {1, 2, 3, 4};
	
	oss << "f + g = " << f + g << endl;
	oss << "f - g = " << f - g << endl;
	oss << "g - f = " << g - f << endl;

	if (f + g != Polynomial <int> {2, 4, 6, 8, 5})
		return false;
	
	if (f - g != Polynomial <int> {0, 0, 0, 0, 5})
		return false;
	
	if (g - f != Polynomial <int> {0, 0, 0, 0, -5})
		return false;

	return true;
}
