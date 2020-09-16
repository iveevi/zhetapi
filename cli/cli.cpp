// C/C++ headers
#include <iostream>
#include <string>

// Engine headers
#include <barn.hpp>
#include <expression.hpp>

using namespace std;

int main()
{
	zhetapi::Barn <double, int> barn;

	string line;
	while (true) {
		cout << "(zhetapi-cli) ";

		getline(cin, line);

		if (line == "q" || line == "quit")
			return 0;

		cout << "\t" << zhetapi::expr_str <double, int> (line) << endl;
	}
}
