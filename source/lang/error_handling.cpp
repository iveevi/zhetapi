#include <lang/error_handling.hpp>

namespace zhetapi {

static const size_t threshold = 5;

size_t levenshtein(const std::string &a, const std::string &b)
{
	using namespace std;
	size_t rows = a.length() + 1;
	size_t cols = b.length() + 1;

	// Make the matrix
	Matrix <size_t> mat(rows, cols,
		[&](size_t i, size_t j) -> size_t {
			if (min(i, j)) {
				return min(
					mat[i - 1][j] + 1,
					min(
						mat[i][j - 1] + 1,
						mat[i - 1][j - 1] + (a[i] == b[j])
					)
				);
			}
			
			return max(i, j);
		}
	);

	return mat[rows - 1][cols - 1];
}

}