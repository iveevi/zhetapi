#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

enum type {
	t_ident,
	t_equals,
	t_expr
};

std::string strtypes[] {
	"ident",
	"equals",
	"expr"
};

struct token {
	type kind;
	std::string name;
};

token code(const std::string &str)
{
	if (str == "=")
		return {t_equals, ""};

	return {t_ident, str};
}

std::ostream &operator<<(std::ostream &os, const token &tok)
{
	std::string out = "[" + std::string(strtypes[tok.kind]) + "]";

	if (tok.kind == t_ident)
		out += " = " + tok.name;

	os << out;

	return out;
}
	
void parse_file(ifstream &fin)
{
	std::vector <std::string> tokens;
	std::string str;

	token t;

	char c;
	
	std::string line;
	while (std::getline(fin, line)) {
		std::istringstream iss(line);

		tokens.clear();
		str.clear();

		while (iss.get(c)) {
			if (isspace(c)) {
				if (!str.empty()) {
					tokens.push_back(str);
					str.clear();
				}
			} else {
				switch (c) {
				case '=': case '(':
				case ')': case ',':
					if (!str.empty()) {
						tokens.push_back(str);
						str.clear();
					}

					tokens.push_back(std::string(1, c));
					break;
				default:
					str += c;
					break;
				}
			}
		}

		if (!str.empty())
			tokens.push_back(str);
	}
}

int main()
{
	ifstream fin("sample");

	parse_file(fin);
}
