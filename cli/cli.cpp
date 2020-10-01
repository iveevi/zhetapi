// C/C++ headers
#include <iostream>
#include <string>
#include <sstream>

// Engine headers
#include <function.hpp>
#include <expression.hpp>

using namespace std;

typedef zhetapi::operand <int> z;
typedef zhetapi::operand <Rational <int>> q;
typedef zhetapi::operand <double> r;

// Global variables
zhetapi::Barn <double, int> barn;

// Inserting variables into the barn
void insert(const string &name, const string &expr)
{
	zhetapi::node_manager <double, int> mgr(expr, barn);

	zhetapi::token *tptr = mgr.value();

	z *o_z = dynamic_cast <z *> (tptr);
	if (o_z) {
		barn.put(name, o_z->get());
		cout << "\t" << tptr->str() << endl;
		return;
	}
	
	q *o_q = dynamic_cast <q *> (tptr);
	if (o_q) {
		barn.put(name, o_q->get());
		cout << "\t" << tptr->str() << endl;
		return;
	}
	
	r *o_r = dynamic_cast <r *> (tptr);
	if (o_r) {
		barn.put(name, o_r->get());
		cout << "\t" << tptr->str() << endl;
		return;
	}
}

void parse(const string &str)
{
	size_t count;
	size_t index;

	count = 0;
	for (size_t i = 0; i < str.size(); i++) {
		if (str[i] == '=') {
			++count;

			index = i;
		}
	}
	
	if (count == 0) {
		try {
			cout << "\t" << zhetapi::expr_str <double, int> (str, barn) << endl;
		} catch (zhetapi::node_manager <double, int> ::undefined_symbol e) {
			cout << "\t" << e.what() << endl;
		}
	} else if (count == 1) {
		try {
			zhetapi::Function <double, int> tmp = std::string(str);

			barn.put(tmp);

			return;
		} catch(...) {}

		string var = str.substr(0, index);
		string expr = str.substr(index + 1);

		istringstream iss(var);

		iss >> var;

		insert(var, expr);
	} else {
		cout << "\tError: invalid syntax" << endl;
	}
}

int main()
{
	barn.put("e", exp(1));
	barn.put("pi", acos(-1));

	string line;
	while (true) {
		cout << "(zhetapi-cli) ";

		getline(cin, line);

		if (line == "q" || line == "quit")
			return 0;

		parse(line);

		zhetapi::Function <double, int> ::barn = barn;
	}
}
