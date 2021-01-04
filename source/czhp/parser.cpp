#include "global.hpp"

#define __lineup(c) 						\
	if (c == '\n') {					\
		fseek(stdin, -2, SEEK_CUR);			\
								\
		c = getchar();					\
		cout << "newline after '" << c << "'" << endl;	\
		c = getchar();					\
	}

// Global scope code
vector <string> global;

static int parse_parenthesized(string &parenthesized)
{
	char c;

	while (isspace(c = getchar())) {
		__lineup(c);
	}

	if (c != '(')
		return -1;

	while ((c = getchar()) != ')')
		parenthesized += c;

	return 0;
}

static int parse_block(string &block)
{
	char c;
	
	while (isspace(c = getchar())) {
		__lineup(c);
	}

	if (c == '{') {
		while ((c = getchar()) != '}') {
			block += c;

			__lineup(c);
		}
	} else {
		fseek(stdin, -1, SEEK_CUR);

		while ((c = getchar()) != '\n' && (c != EOF)) {
			block += c;
		}

		__lineup(c);
	}

	return 0;
}

static void check(string &keyword)
{
	if (keyword == "if") {
		cout << "IF" << endl;
		string parenthesized;
		
		if (parse_parenthesized(parenthesized)) {
			printf("Syntax error at line %lu: missing parenthesis after an if\n", line);
		} else {
			cout << "\tparen = \"" << parenthesized << "\"" << endl;
		}

		string block;
		
		if (parse_block(block)) {
			printf("Syntax error at line %lu: missing statement after if\n", line);
		} else {
			cout << "\tblock = \"" << block << "\"" << endl;
		}
		
		keyword.clear();
	}

	if (keyword == "for")
		cout << "FOR" << endl;
}

// Parsing machine
int parse()
{
	string tmp;

	char c;
	while ((c = getchar()) != EOF) {
		if (!isspace(c))
			tmp += c;

		__lineup(c);

		// cout << "tmp = " << tmp << endl;
		check(tmp);
	}

	return 0;
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
