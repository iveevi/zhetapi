#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>

#include <matrix.h>
#include <vector.h>
#include <rational.h>

using namespace std;

Matrix <Rational <int>> vandermonde(vector <pair <Rational <int>, Rational <int>>> S, size_t deg)
{
	return Matrix <Rational <int>> {S.size(), deg + 1, [&](size_t i, size_t j) {
		Rational <int> t = 1;

		for (size_t k = 0; k < deg - j; k++)
			t *= S[i].first;

		return t;
	}};
}

// Copy instead of pass by reference
Matrix <Rational <int>> extended_gauss_jordan(Matrix <Rational <int>> A)
{
	size_t n = A.get_rows();

	Rational <int> k;
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
			
			// A.add_rows(j, i, -k);
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
pair <Rational <int>, Rational <int>> find_range(Vector <Rational <int>> Qa, Vector <Rational <int>> c)
{
	size_t rows = Qa.size();

	Rational <int> mn = {-1, 0};
	Rational <int> mx = {1, 0};

	for (size_t i = 0; i < rows; i++) {
		if (Qa[i] < Rational <int> {0, 1})
			mx = min(mx, c[i]/Qa[i]);
		else
			mn = max(mn, c[i]/Qa[i]);
	}

	return {mn, mx};
}

Rational <int> optimum(Vector <Rational <int>> Qa, Vector <Rational <int>> c)
{
	return inner(Qa, c)/inner(Qa, Qa);
}

int main()
{
	// Matrix <Rational <int>> ::nice = true;

	vector <pair <Rational <int>, Rational <int>>> D {
		{0, 0},
		{2, 4},
		{6, 2},
		{9, 9},
		{12, 3},
		{16, 4}
	};

	vector <pair <Rational <int>, Rational <int>>> Sf {
		{2, 4},
		{9, 9},
		{16, 4}
	};

	auto y = Vector <Rational <int>> {D.size(), [&](size_t i) {
		return D[i].second;
	}};

	auto y_hat = Vector <Rational <int>> {Sf.size(), [&](size_t i) {
		return Sf[i].second;
	}};

	auto P_D = vandermonde(D, Sf.size());
	auto P_Sf = vandermonde(Sf, Sf.size());

	auto P_Sf_rest = P_Sf.slice({0, 1}, {2, Sf.size()});

	auto P_Sf_first = (Rational <int> (-1)) * P_Sf.get_column(0);

	auto S_pre = P_Sf_first.append_right(y_hat);

	auto Aug = P_Sf_rest.append_right(S_pre);

	auto I_Aug = extended_gauss_jordan(Aug);

	auto S = (I_Aug.slice({0, Sf.size()}, {I_Aug.get_rows() - 1, 
				I_Aug.get_cols() - 1})).append_above({{1, 0}});

	auto Q = P_D * S;

	auto Qa = Q.get_column(0);
	auto Qb = Q.get_column(1);

	auto c = y - Qb;


	pair <Rational <int>, Rational <int>> range = find_range(Qa, c);

	Rational <int> lambda = optimum(Qa, c);

	Rational <int> p = 0;

	if (range.first.is_inf()) {
		p = range.second;
	} else if (range.second.is_inf()) {
		p = range.first;
	} else {
		if (abs(range.first - lambda) < abs(range.second - lambda)) {
			p = range.first;
		} else {
			p = range.second;
		}
	}

	auto a = Vector <Rational <int>> {{p}, {1}};

	auto params = S * a;

	cout << "params: " << params <<  endl;
}
