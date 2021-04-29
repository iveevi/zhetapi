#include <cstdio>

#include <engine.hpp>
#include <lang/error_handling.hpp>

namespace zhetapi {

static const size_t threshold = 3;

size_t levenshtein(const std::string &a, const std::string &b)
{
	size_t rows = a.length() + 1;
	size_t cols = b.length() + 1;

	// Make the matrix
	Matrix <size_t> mat(rows, cols,
		[&](size_t i, size_t j) -> size_t {
			return std::min(i, j) ? 0 : std::max(i, j);
		}
	);

	for (size_t i = 1; i < rows; i++) {
		for (size_t j = 1; j < cols; j++) {
			mat[i][j] = std::min(
				mat[i - 1][j] + 1,
				std::min(
					mat[i][j - 1] + 1,
					mat[i - 1][j - 1] + (a[i - 1] != b[j - 1])
				)
			);
		}
	}

	return mat[rows - 1][cols - 1];
}

Args symbol_suggestions(const std::string &symbol, const Args &choices)
{
	Args valid;

	for (const std::string &str : choices) {
		if (levenshtein(str, symbol) < threshold)
			valid.push_back(str);
	}

	return valid;
}

void symbol_error_msg(const std::string &symbol, Engine *context)
{
	// Add a function for classifying the error
	std::fprintf(stderr, "Error: The symbol %s has not been defined yet.",
			symbol.c_str());
	
	Args suggs = symbol_suggestions(symbol, context->symbol_list());

	size_t n = suggs.size();
	if (n) {
		std::fprintf(stderr, "\nSuggested symbols: {");

		for (size_t i = 0; i < n; i++) {
			std::fprintf(stderr, "%s%s", suggs[i].c_str(),
					(i < n - 1) ? ", " : "");
		}
		
		std::fprintf(stderr, "}?\n");
	}
}

}