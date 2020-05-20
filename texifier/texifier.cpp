#include <iostream>
#include <fstream>
#include <string>

using namespace std;

string convert(string line)
{
	string out;

	char c;
	for (int i = 0; i < line.length(); i++) {
		c = line[i];

		if (isalnum(c)) {
			out += c;
			continue;
		}

		switch (c) {
		case '(':
			out += "\\left(";
			break;
		case ')':
			out += "\\right)";
			break;
		case '=':
		case ',':
		default:
			out += c;
		}
	}

	return out;
}

int main()
{
	ofstream fout("../build/texifier.out");
	ifstream fin("../build/texifier.in");
	
	string input;

	getline(fin, input);

#ifdef DEBUG
	cout << "[Texifier]: received \"" << input << "\"" << endl;
#endif

	fout << convert(input) << endl;

#ifdef DEBUG
	cout << "[Texifier]: returning \"" << convert(input) << "\"" << endl;
#endif

	return 0;
}
