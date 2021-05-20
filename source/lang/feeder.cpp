#include <lang/feeder.hpp>

namespace zhetapi {

bool is_terminal(char c)
{
	return (c == '\0')
		|| (c == EOF);
}

// Feeder functions
void Feeder::skip_line()
{
	char c;
	while ((c = feed()) != EOF) {
		if (c == '\n')
			break;
	}
}

// TODO: optimize while keeping a cyclic buffer
void Feeder::skip_until(const std::string &str)
{
	std::string tmp;
	char c;

	for (size_t i = 0; i < str.length(); i++) {
		if ((c = feed()) == EOF)
			return;

		tmp += c;
	}

	while (true) {
		if (tmp == str)
			break;

		if ((c = feed()) == EOF)
			break;

		tmp = tmp.substr(1) + c;
	}
}

std::string Feeder::extract_quote()
{
	std::string out;
	char c;

	while ((c = feed()) != EOF) {
		if (c == '\\' && peek() == '\"') {
			feed();

			out += "\\\"";
		} else {
			if (c == '\"')
				break;

			out += c;
		}
	}

	return out;
}

std::string Feeder::extract_parenthesized()
{
	std::string out;
	char c;

	while ((c = feed()) != EOF) {
		if (c == ')')
			break;
		
		out += c;
	}

	return out;
}

}
