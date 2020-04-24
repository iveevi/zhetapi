#include <iostream>
#include <fstream>

#include "expression.h"
#include "var_stack.h"

using namespace std;

/* Shared files */
ofstream fout("../build/driver.out");
ifstream fin("../build/driver.in");

/* Structures with state of application */
vector <string> history;

var_stack <double> vst;

vector <variable <double>> vals {
	variable <double> {"pi", acos(-1)},
	variable <double> {"e", exp(1)}
};

extern "C" {
	void load()
	{
		for (variable <double> v : vals)
			vst.insert(v);
	}

	string process(string in)
	{
		try {
			fout << expression <double> ::in_place_evaluate(in, vst) << endl;
		} catch(node <double> ::undefined_symbol e) {
			fout << "\\text{Could not identify symbol or variable} $"
				<< e.what() << "$ \\text{ [Undefined Symbol Error].}" << endl;
		} catch (...) {
			fout << "\\text{Could not evaluate expression [Unkown Error].}" << endl;
		}
	}

	string bar()
	{
		return "Hello World";
	}
}
