// C/C++ headers
#include <iostream>
#include <vector>

// Engine headers
#include <function.hpp>

// Includes
using namespace std;

// Global scope code
vector <string> global;

// Zhetapi API storage
zhetapi::Barn <double, int> barn;

// Parsing states
int block = 0;

// Parsing machine
void parse(char c)
{
	static string tmp;

	if (c == '\t')
		return;

	if (c == '{') {
		block++;
		cout << "\tBlock Enter" << endl;
	}

	if (!block) {
		if (c == '\n' || c == ',') {
			if (!tmp.empty()) {
				global.push_back(tmp);

				tmp.clear();
			}
		} else {
			if (!isspace(c))
				tmp += c;
		}
	}
	
	if (c == '}') {
		block--;
		cout << "\tBlock Leave" << endl;
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
	cout << "================================" << endl;
	cout << "processing [" << statement << "]" << endl;

	vector <string> tmp = split(statement);
	
	size_t tsize = tmp.size();
	if (tsize > 1) {
		zhetapi::node_manager <double, int> mg(tmp[tsize - 1], barn);

		cout << "\ntree:" << endl;
		mg.print();

		zhetapi::Token *tptr = mg.value();
		for (int i = tsize - 2; i >= 0; i--)
			barn.put(tptr, tmp[i]);
		
		cout << "\npost-state:" << endl;
		barn.print();
		
		delete tptr;
	}

	// All functions and algorithms are stored in barn
	zhetapi::node_manager <double, int> mg(statement, barn);

	cout << "tree: " << endl;
	mg.print();

	// "Execute" the statement
	mg.value();
}

zhetapi::Token *print(const std::vector <zhetapi::Token *> &ins)
{
	for (zhetapi::Token *tptr : ins)
		cout << tptr->str();
	
	return ins[0];
}

// Main
int main()
{
	// Barn setup	
	barn.put(zhetapi::Registrable("print", &print));

	// Input
	char c;

	while (cin.get(c))
		parse(c);
	
	// Flush parse
	parse('\n');

	cout << "global:" << endl;
	for (string str : global)
		cout << "\t" << str << endl;
	
	for (string statement : global)
		process(statement);
}
