#include "../../engine/lang/feeder.hpp"

namespace zhetapi {

bool is_terminal(char c)
{
	return (c == '\0')
		|| (c == EOF);
}

// Feeder functions
Feeder::Feeder(const std::string &loc) : _location(loc) {}

const std::string &Feeder::location() const
{
	return _location;
}

void Feeder::set_end(char c, int count)
{
	this->set_end({c, count});
}

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

	// TODO: get line number at the start
	size_t line = this->line();
	bool warned = false;

	while ((c = feed()) != EOF) {
		if (c == '\\' && peek() == '\"') {
			feed();

			out += "\\\"";
		} else {
			if (c == '\"')
				break;
			
			// Don't warn multiple times
			if (c == '\n' && !warned) {
				warned = true;
				std::cout << detail::warning
					<< " string literal at "
					<< location() << ":" << line
					<< " is unterminated in"
					<< " that line." << std::endl;
			}

			out += c;
		}
	}

	return out;
}

std::string Feeder::extract_parenthesized()
{
	std::string out;
	char c;

	int level = 1;
	while ((c = feed()) != EOF) {
		if (c == ')') {
			if (level == 1)
				break;

			level--;
		} else if (c == '(') {
			// TODO: throw if below 0
			level++;
		}

		out += c;
	}

	return out;
}

std::pair <std::string, Args> Feeder::extract_signature()
{
	std::string ident;
	std::string tmp;
	Args args;

	char c;
	while (isspace(c = feed()));

	ident = c;
	while ((c = feed()) != '(')
		ident += c;

	while ((c = feed()) != ')') {
		if (c == ',') {
			if (!tmp.empty()) {
				args.push_back(tmp);

				tmp.clear();
			}
		} else if (!isspace(c)) {
			tmp += c;
		}
	}

	if (!tmp.empty())
		args.push_back(tmp);

	return {ident, args};
}

}
