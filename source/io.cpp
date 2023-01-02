// Standard headers
#include <cassert>
#include <iomanip>
#include <sstream>

// Library headers
#include "../include/io/print.hpp"

namespace zhetapi {

namespace io {

// TODO: create a separate (single header) library for this
std::string table(const Args &headers,
		const std::vector <Args> &rows)
{
	assert(headers.size() == rows[0].size());

	// Special characters
	std::string vert = "\u2502";
	std::string horiz = "\u2500";

	// Top corners
	std::string tl = "\u250C";
	std::string tr = "\u2510";

	// Bottom corners
	std::string bl = "\u2514";
	std::string br = "\u2518";

	// Calculate column widths
	std::vector <size_t> widths(headers.size(), 0);
	for (int i = 0; i < headers.size(); i++)
		widths[i] = headers[i].length();

	for (const auto &row : rows) {
		for (size_t i = 0; i < row.size(); i++)
			widths[i] = std::max(widths[i], row[i].size());
	}

	// Stream
	std::stringstream ss;
	
	// Print the top
	ss << tl;
	for (size_t i = 0; i < headers.size(); i++) {
		for (int n = 0; n < widths[i] + 2; n++)
			ss << horiz;

		if (i < headers.size() - 1)
			ss << "\u252C";
		else
			ss << tr;
	}
	ss << "\n";
	
	// Print the header
	for (int i = 0; i < headers.size(); i++) {
		ss << vert << " " << std::setw(widths[i])
			<< headers[i] << " ";
	}
	ss << vert << "\n";

	// Post header separator
	ss << "\u251C";
	for (size_t i = 0; i < headers.size(); i++) {
		for (int n = 0; n < widths[i] + 2; n++)
			ss << horiz;

		if (i < headers.size() - 1)
			ss << "\u253C";
		else
			ss << "\u2524";
	}
	ss << "\n";

	// Print the rows
	for (const auto &row : rows) {
		for (int i = 0; i < row.size(); i++) {
			ss << vert << " " << std::setw(widths[i])
				<< row[i] << " ";
		}
		ss << vert << std::endl;
	}

	// Post row separator
	ss << bl;
	for (size_t i = 0; i < headers.size(); i++) {
		for (int n = 0; n < widths[i] + 2; n++)
			ss << horiz;

		if (i < headers.size() - 1)
			ss << "\u2534";
		else
			ss << br;
	}
	ss << "\n";

	return ss.str();
}

}

}
