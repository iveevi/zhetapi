#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>

#include <matrix.h>
#include <vector.h>

using namespace std;

Matrix <double> vandermonde(vector <pair <double, double>> S, size_t deg)
{
	return Matrix <double> {S.size(), deg + 1, [&](size_t i, size_t j) {
		return pow(S[i].first, deg - j);
	}};
}

// Copy instead of pass by reference
Matrix <double> extended_gauss_jordan(Matrix <double> A)
{
	size_t n = A.get_rows();

	double k;
	for (size_t i = 0; i < n; i++) {
		for (size_t j = i + 1; j < n; j++) {
			k = A[j][i]/A[i][i];

			A.add_rows(j, i, -k);
		}
	}

	for (int i = n - 1; i > -1; i--) {
		for (int j = i - 1; j > -1; j--) {
			k = A[j][i]/A[i][i];
			
			A.add_rows(j, i, -k);
		}
	}

	for (size_t i = 0; i < n; i++)
		A.multiply_row(i, 1/A[i][i]);

	return A;
}

// Max fitting (>)
pair <double, double> find_range(Vector <double> Qa, Vector <double> c)
{
	size_t rows = Qa.size();

	double mn = -numeric_limits <double> ::infinity();
	double mx = numeric_limits <double> ::infinity();

	for (size_t i = 0; i < rows; i++) {
		if (Qa[i] < 0) {
			mx = min(mx, c[i]/Qa[i]);
		} else {
			mn = max(mn, c[i]/Qa[i]);
		}
	}

	return {mn, mx};
}

double optimum(Vector <double> Qa, Vector <double> c)
{
	return inner(Qa, c)/inner(Qa, Qa);
}

int main()
{
	Matrix <double> ::nice = true;

	vector <pair <double, double>> D {
		{0, 0},
		{2, 4},
		{6, 2},
		{9, 9},
		{12, 3},
		{16, 4}
	};

	vector <pair <double, double>> Sf {
		{2, 4},
		{9, 9},
		{16, 4}
	};

	auto y = Vector <double> {D.size(), [&](size_t i) {
		return D[i].second;
	}};

	auto y_hat = Vector <double> {Sf.size(), [&](size_t i) {
		return Sf[i].second;
	}};

	auto P_D = vandermonde(D, Sf.size());
	auto P_Sf = vandermonde(Sf, Sf.size());

	auto P_Sf_rest = P_Sf.slice({0, 1}, {2, Sf.size()});

	auto P_Sf_first = -1.0 * P_Sf.get_column(0);

	cout << "P:\n" << P_D << endl;
	cout << "P':\n" << P_Sf << endl;

	cout << "y:\n" << y << endl;
	cout << "(y):\n" << y_hat << endl;

	cout << "P'(+1):\n" << P_Sf_first << endl;
	cout << "P'(-1):\n" << P_Sf_rest << endl;

	auto S_pre = P_Sf_first.append_right(y_hat);

	cout << "S':\n" << S_pre << endl;

	auto Aug = P_Sf_rest.append_right(S_pre);

	cout << "Aug:\n" << Aug << endl;

	auto I_Aug = extended_gauss_jordan(Aug);

	cout << "I_Aug:\n" << I_Aug << endl;

	auto S = (I_Aug.slice({0, Sf.size()}, {I_Aug.get_rows() - 1, 
				I_Aug.get_cols() - 1})).append_above({{0, 1}});

	cout << "S:\n" << S << endl;

	auto Q = P_D * S;

	auto Qa = Q.get_column(0);
	auto Qb = Q.get_column(1);

	auto c = y - Qb;

	cout << "Q:\n" << Q << endl;

	pair <double, double> range = find_range(Qa, c);

	cout << "range-min: " << range.first << endl;
	cout << "range-max: " << range.second << endl;

	double lambda = optimum(Qa, c);

	cout << "lambda: " << lambda << endl;

	double p = 0;
	if (fabs(range.first - lambda) < fabs(range.second - lambda))
		p = range.first;
	else
		p = range.second;

	cout << "optimal p: " << p << endl;

	auto a = Vector <double> {{p}, {1}};

	auto params = S * a;

	cout << "params: " << params <<  endl;
}
