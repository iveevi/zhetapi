#include <iostream>
#include <fstream>

#include "expression.h"
#include "var_stack.h"

using namespace std;

ofstream fout("../build/driver.out");
ifstream fin("../build/driver.in");

int main()
{
	var_stack <double> vst;

	vector <variable <double>> vals {
		variable <double> {"pi", acos(-1)},
		variable <double> {"e", exp(1)}
	};
	
	for (variable <double> v : vals)
		vst.insert(v);

	string line;
	
	fin >> line;

#ifdef DEBUG
	cout << "[driver]: received \"" << line << "\"" << endl;
#endif

	try {
		//fout << "\\text{" << line << "}" << endl;
		//fout << "\\[" << expression <double> ::in_place_evaluate(line) << "\\]" << endl;
		fout << expression <double> ::in_place_evaluate(line, vst) << endl;
	} catch (...) {
		fout << "Could not evaluate expression" << endl;
	}
}
