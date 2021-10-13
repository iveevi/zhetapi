#include "port.hpp"

TEST(diag_matrix)
{
	using namespace zhetapi;
	using namespace zhetapi::linalg;

	Matrix <int> A {
		{1, 0, 0, 0},
		{0, 2, 0, 0},
		{0, 0, 3, 0},
		{0, 0, 0, 4}
	};

	vector <int> cs {1, 2, 3, 4};

	Matrix <int> B = diag(cs);
	Matrix <int> C = diag(1, 2, 3, 4);

	oss << "A = " << A << endl;
	oss << "B = " << B << endl;
	oss << "C = " << C << endl;

	if ((A != B) || (A != C) || (B != C)) {
		oss << "Not equal..." << endl;

		return false;
	}

	return true;
}

TEST(qr_decomp)
{
	using namespace zhetapi;
	using namespace zhetapi::linalg;

	Matrix <double> A = diag(1.0, 4.0, 1.0, 5.0);

	oss << "A = " << A << endl;

	auto qr = qr_decompose(A);

	oss << "\tQ = " << qr.q() << endl;
	oss << "\tR = " << qr.r() << endl;
	oss << "\tQR = " << qr.product() << endl;

	oss << "Error = " << (qr.product() - A).norm() << endl;

	if ((qr.product() - A).norm() > 1e-10) {
		oss << "Failure: A != QR" << endl;

		return false;
	}

	return true;
}

TEST(lq_decomp)
{
	using namespace zhetapi;
	using namespace zhetapi::linalg;

	Matrix <double> A = diag(1.0, 4.0, 1.0, 5.0);

	oss << "A = " << A << endl;

	auto lq = lq_decompose(A);

	oss << "\tL = " << lq.l() << endl;
	oss << "\tQ = " << lq.q() << endl;
	oss << "\tLQ = " << lq.product() << endl;

	oss << "Error = " << (lq.product() - A).norm() << endl;

	if ((lq.product() - A).norm() > 1e-10) {
		oss << "Failure: A != LQ" << endl;

		return false;
	}

	return true;
}

TEST(qr_alg)
{
	using namespace zhetapi;
	using namespace zhetapi::linalg;

	// Test on diagonal matrices:
	// eigenvalues should be equal to
	// the diagonal entries
	Matrix <double> A = diag(1.0, 2.0, 3.0, 4.0);

	oss << "A = " << A << endl;

	Vector <double> E = qr_algorithm(A);

	oss << "E = " << E << endl;

	for (size_t i = 0; i < E.size(); i++) {
		if (E[i] != A[i][i]) {
			oss << "Incorrect eigenvalues..." << endl;

			return false;
		}
	}

	// Test on Fibonacci matrix
	A = {{1, 1}, {1, 0}};

	E = qr_algorithm(A);

	Vector <double> G {
		(double) (1 + sqrt(5.0))/2.0,
		(double) (1 - sqrt(5.0))/2.0
	};

	oss << "Fib. matrix = " << A << endl;
	oss << "E = " << E << endl;
	oss << "G = " << G << endl;

	oss << "\nError = " << (E - G).norm() << endl;

	return true;
}

TEST(matrix_props)
{
	using namespace zhetapi;
	using namespace zhetapi::linalg;

	Matrix <double> A = diag(1.0, 2.0, 3.0, 4.0);
	Matrix <double> I = Matrix <double> ::identity(4);

	auto qr = qr_decompose(A);
	auto lq = lq_decompose(A);

	oss << "Is A diagonal? " << (is_diagonal(A) ? "yes" : "no") << endl;
	if (!is_diagonal(A)) {
		oss << "\tWrong answer..." << endl;

		return false;
	}

	oss << "Is I identity? " << (is_identity(I) ? "yes" : "no") << endl;
	if (!is_identity(I)) {
		oss << "\tWrong answer..." << endl;

		return false;
	}

	oss << "Is Q orthogonal? " << (is_orthogonal(qr.q()) ? "yes" : "no") << endl;
	if (!is_orthogonal(qr.q())) {
		oss << "\tWrong answer..." << endl;

		return false;
	}

	oss << "Is R upper triangular? " << (is_upper_triangular(qr.r()) ? "yes" : "no") << endl;
	if (!is_upper_triangular(qr.r())) {
		oss << "\tWrong answer..." << endl;

		return false;
	}

	oss << "Is L lower triangular? " << (is_lower_triangular(lq.l()) ? "yes" : "no") << endl;
	if (!is_lower_triangular(lq.l())) {
		oss << "\tWrong answer..." << endl;

		return false;
	}

	return true;
}
