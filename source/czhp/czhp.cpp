// Source headers
#include "global.hpp"

// Global scope code
vector <string> global;

// Zhetapi API storage
zhetapi::Barn <double, int> barn;

// Parsing machine
void parse(char c)
{
	static string tmp;

	static int block = 0;
	static int paren = 0;

	static bool quote = false;

	switch (c) {
	case '{':
		cout << "\tBlock Enter" << endl;
		block++;

		break;
	case '}':
		cout << "\tBlock Leave" << endl;
		block--;

		break;
	case '(':
		tmp += c;

		paren++;

		break;
	case ')':
		tmp += c;

		paren--;

		break;
	case '\"':
		quote = !quote;

		tmp += c;

		break;
	case ',':
		if (!tmp.empty() && !paren) {
			global.push_back(tmp);

			tmp.clear();
		} else {
			tmp += c;
		}

		break;
	case '\n':
		if (!tmp.empty()) {
			global.push_back(tmp);

			tmp.clear();
		}

		break;
	default:
		if (quote || !isspace(c))
			tmp += c;
		
		break;
	}
}

// Splitting equalities
vector <string> split(string str)
{
	vector <string> out;
	size_t n;

	n = str.length();

	string tmp;
	for (size_t i = 0; i < n; i++) {
		if (str[i] == '=') {
			if (!tmp.empty()) {
				out.push_back(tmp);

				tmp.clear();
			}
		} else {
			tmp += str[i];
		}
	}

	if (!tmp.empty())
		out.push_back(tmp);

	return out;
}

// Global processing
void process(string statement)
{
	// Skip comments
	if (statement[0] == '#')
		return;

	vector <string> tmp = split(statement);
	
	size_t tsize = tmp.size();
	if (tsize > 1) {
		zhetapi::Token *tptr = nullptr;
		
		try {
			zhetapi::node_manager <double, int> mg(tmp[tsize - 1], barn);

			tptr = mg.value();
		} catch (...) {}

		for (int i = tsize - 2; i >= 0; i--) {
			string ftr = tmp[i] + " = " + tmp[tsize - 1];

			try {
				zhetapi::Function <double, int> f = ftr;

				barn.put(f);
			} catch (...) {
				barn.put(tptr, tmp[i]);
			}
		}
		
		delete tptr;
	} else {
		// All functions and algorithms are stored in barn
		zhetapi::node_manager <double, int> mg(statement, barn);

		// "Execute" the statement
		mg.value();
	}
}

// Main
int main(int argc, char *argv[])
{
	if (argc == 2) {
		if (!freopen(argv[1], "r", stdin)) {
			printf("Fatal error: failed to open file '%s'.\n", argv[1]);

			exit(-1);
		}
	}
	
	// Barn setup	
	barn.put(zhetapi::Registrable("print", &print));
	barn.put(zhetapi::Registrable("println", &println));

	// Input
	char c;

	while (cin.get(c))
		parse(c);
	
	// Flush parse
	parse('\n');
	for (string statement : global)
		process(statement);
}
