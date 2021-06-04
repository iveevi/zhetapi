#include <linalg.hpp>

namespace zhetapi {

namespace linalg {

const long double GAMMA = 1.15470053837925152901 + 1e-4;
const long double EPSILON = 1e-10;

static Mat size_reduce(const Mat &H)
{
	// Dimensions
	size_t n = H.get_rows();

	// Unimodular matrix
	Mat D = Mat::identity(n);

	for (size_t i = 1; i < n; i++) {
		for (int j = i - 1; j >= 0; j--) {
			long double q = std::floor(0.5 + H[i][j]/H[j][j]);

			for (size_t k = 0; k < n; k++)
				D[i][k] -= q * D[j][k];
		}
	}

	return D;
}

static std::pair <size_t, Mat> bergman_swap(const Mat &H, long double gamma)
{
	// Dimensions
	size_t n = H.get_rows();

	// Unimodular matrix
	Mat D = Mat::identity(n);

	long double max = 0;

	size_t r = -1;
	for (size_t i = 0; i < n - 1; i++) {
		long double tmp = pow(gamma, i) * fabs(H[i][i]);

		if (tmp > max) {
			max = tmp;

			r = i;
		}
	}

	D.swap_rows(r, r + 1);

	return {r, D};
}

static Mat corner(const Mat &H, size_t r)
{
	// Dimensions
	size_t n = H.get_rows();

	// Cached constants
	long double eta = H[r][r];
	long double beta = H[r + 1][r];
	long double lambda = H[r + 1][r + 1];
	long double delta = sqrt(beta * beta + lambda * lambda);

	// Orthonal matrix Q
	return Mat(n - 1, n - 1,
		[&](size_t i, size_t j) -> long double {
			if ((i == j)) {
				if ((i == r) || (i == r + 1))
					return beta/delta;
				else
					return 1;
			} else if ((i == r) && (j == r + 1)) {
				return -lambda/delta;
			} else if ((i == r + 1) && (j == r)) {
				return lambda/delta;
			}

			return 0;
		}
	);
}

// Using the PSLQe algorithm from https://arxiv.org/abs/1707.05037
Vec pslq(const Vec &a, long double gamma, long double epsilon)
{
	// Length of a
	size_t n = a.size();

	// Save a copy of a (take normalized value)
	Mat alpha = a.normalized().transpose();

	// Partial sums
	Vec s(n,
		[&](size_t j) -> long double {
			long double sum = 0;

			for (size_t k = j; k < n; k++)
				sum += alpha[0][k] * alpha[0][k];

			return sqrt(sum);
		}
	);

	// Construct the matrix H_alpha
	Mat H_alpha(n, n - 1,
		[&](size_t i, size_t j) -> long double {
			if ((i < j) && (j < n - 1))
				return 0;
			else if ((i == j) && (i < n - 1))
				return s[i + 1]/s[i];

			return -(alpha[0][i] * alpha[0][j])/(s[j] * s[j + 1]);
		}
	);

	Mat H = H_alpha;

	Mat A = Mat::identity(n);
	Mat B = Mat::identity(n);

	Mat D = size_reduce(H);

	// Update lambda: returns false if H has a zero on the diagonal
	auto update = [&]() -> bool {
		Mat D_inv = D.inverse();

		alpha = alpha * D_inv;
		H = D * H;
		A = D * A;
		B = B * D_inv;

		// Check diagonal elements for non-zero
		for (size_t i = 0; i < H.get_cols(); i++) {
			if (H[i][i] < epsilon)
				return false;
		}

		return true;
	};

	// Update once first
	update();

	// Main loop
	while (fabs(H[n - 1][n - 2]) >= epsilon) {
		auto Dr = bergman_swap(H, gamma);

		D = Dr.second;

		if (!update())
			break;

		if (Dr.first < n - 2)
			H *= corner(H, Dr.first);

		D = size_reduce(H);

		if (!update())
			break;
	}

	return B.get_column(n - 2);
}

}

}
