// C/C++ headers
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>

// Engine headers
#include <matrix.h>
#include <vector.h>
#include <rational.h>

#define INF numeric_limits <double> ::infinity()

using namespace std;

typedef vector <pair <double, double>> DataSet;

typedef Vector <double> Vec;
typedef Matrix <double> Mat;

typedef pair <double, double> Range;

typedef double Scalar;

Mat vandermonde(DataSet S, size_t deg)
{
	return Mat {S.size(), deg + 1, [&](size_t i, size_t j) {
		Scalar t = 1;

		for (size_t k = 0; k < deg - j; k++)
			t *= S[i].first;

		return t;
	}};
}

Mat extended_gauss_jordan(Mat A)
{
	size_t n = A.get_rows();

	Scalar k;
	for (size_t i = 0; i < n; i++) {
		for (size_t j = i + 1; j < n; j++) {
			k = A[j][i]/A[i][i];

			for (size_t q = 0; q < A.get_cols(); q++)
				A[j][q] -= k * A[i][q];
		}
	}

	for (int i = n - 1; i > -1; i--) {
		for (int j = i - 1; j > -1; j--) {
			k = A[j][i]/A[i][i];

			for (size_t q = 0; q < A.get_cols(); q++)
				A[j][q] -= k * A[i][q];
		}
	}

	for (size_t i = 0; i < n; i++) {
		Rational <int> t = A[i][i];

		for (size_t j = 0; j < A.get_cols(); j++)
			A[i][j] /= t;
	}
	
	return A;
}

// Max fitting (>)
Range find_range(Vec Qa, Vec c)
{
	size_t rows = Qa.size();

	Scalar mx = INF;
	Scalar mn = -INF;

	for (size_t i = 0; i < rows; i++) {
		if (Qa[i] < 0)
			mx = min(mx, c[i]/Qa[i]);
		else
			mn = max(mn, c[i]/Qa[i]);
	}

	return {mn, mx};
}

Scalar optimum(Vec Qa, Vec c)
{
	cout << "Qa: " << Qa << endl;
	cout << "c: " << c << endl;
	
	Scalar usum = 0;
	Scalar dsum = 0;

	for (size_t i = 0; i < Qa.size(); i++)
		usum += Qa[i] * c[i];
	
	for (size_t i = 0; i < Qa.size(); i++)
		dsum += Qa[i] * Qa[i];

	// return inner(Qa, c)/inner(Qa, Qa);
	
	return usum/dsum;
}

Scalar error(Vec Qa, Vec c, Scalar value)
{
	Scalar sum = 0;

	for (size_t i = 0; i < Qa.size(); i++) {
		Scalar tmp = Qa[i] * value - c[i];

		sum += tmp * tmp;
	}

	return sum;
}

Vec psa(DataSet D, DataSet Sf)
{
	Vec y {D.size(), [&](size_t i) {
		return D[i].second;
	}};

	Vec y_hat {Sf.size(), [&](size_t i) {
		return Sf[i].second;
	}};

	Mat P_D = vandermonde(D, Sf.size());
	Mat P_Sf = vandermonde(Sf, Sf.size());

	// First column of P_Sf
	Mat P_Sf_first = (Scalar (-1)) * P_Sf.get_column(0);
	Mat P_Sf_rest = P_Sf.slice({0, 1}, {2, Sf.size()});

	Mat S_pre = P_Sf_first.append_right(y_hat);

	Mat Aug = P_Sf_rest.append_right(S_pre);

	Mat I_Aug = extended_gauss_jordan(Aug);

	// Extract S
	Mat S = (I_Aug.slice({0, Sf.size()}, {I_Aug.get_rows() - 1, 
				I_Aug.get_cols() - 1})).append_above({{1, 0}});

	Mat Q = P_D * S;

	Vec Qa = Q.get_column(0);
	Vec Qb = Q.get_column(1);

	Vec c = y - Qb;

	Range range = find_range(Qa, c);

	Scalar lambda = optimum(Qa, c);

	Scalar p = 0;

	if ((range.first == INF) || (range.second == -INF)) {
		p = range.second;
	} else if ((range.second == INF) || (range.second == -INF)) {
		p = range.first;
	} else {
		cout << "lambda: " << lambda << endl;
		cout << "\terr: " << error(Qa, c, lambda) << endl;
		cout << "range: (" << range.first << ", " << range.second << ")" << endl;
		cout << "\ta: " << error(Qa, c, range.first) << endl;
		cout << "\tb: " << error(Qa, c, range.second) << endl;
		// If lambda is in C(p) = [a, b]
		//if ((lambda >= range.first) && (lambda <= range.second)) {
		//	p = lambda;
		// Find the projection of lambda onto C(p)
		// 	(the closest point to lambda in C(p))
		/*} else*/ if (abs(range.first - lambda) < abs(range.second - lambda)) {
			p = range.first;
		} else {
			p = range.second;
		}
	}

	Vec a = Vec {{p}, {1}};

	return S * a;
}

int main()
{
	DataSet D {
		{0, 0},
		{2, 4},
		{6, 2},
		{9, 9},
		{12, 3},
		{16, 4}
	};

	DataSet Sf {
		{2, 4},
		{9, 9},
		{16, 4}
	};

	cout << psa(D, Sf) << endl;
}
