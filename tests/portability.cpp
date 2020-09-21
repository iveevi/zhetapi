// C/C++ headers
#include <iostream>

// Engine headers
#include <barn.hpp>
#include <function.hpp>
#include <engine.hpp>
#include <expression.hpp>
#include <tensor.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	cout << "Running portability tests..." << endl;

	//////////////////////////////////////////////////////////////

	cout << endl << "==========================================================" << endl;
	cout << endl << "Placement into Barns...\n" << endl;

	Barn <double, int> b;

	Variable <double> var {"e", 2.17};

	b.put(var);

	cout << "var: " << var << endl;
	cout << "barn-var: " << b.retrieve <double> ("e") << endl;
	cout << "barn-var: " << b.get("e")->str() << endl;

	//////////////////////////////////////////////////////////////

	cout << endl << "==========================================================" << endl;
	cout << endl << "Expression computation...\n" << endl;

	Barn <double, int> barn;

	barn.put(Variable <double> {"e", exp(1)});
	barn.put(Variable <double> {"pi", acos(-1)});

	std::string str;

	while (getline(cin, str)) {
		try {
			zhetapi::node_manager <double, int> tmp(str, barn);

			zhetapi::token *tptr = tmp.value();

			cout << "Value: " << tptr->str() << endl;
		} catch (Barn <double, int> ::unknown_operation_overload_exception e) {
			cout << e.what() << endl;
		} catch (zhetapi::node_manager <double, int> ::error e) {
			cout << e.what() << endl;
		} catch (...) {
			cout << "Another Exception" << endl;
		}
	}

	//////////////////////////////////////////////////////////////

	cout << endl << "==========================================================" << endl;
	cout << endl << "Function computation and differentiation...\n" << endl;

	Function <double, int> f = std::string("f(x, y) = x^2 + y");

	try {
		cout << "f(10, -4) = " << f(10, -4)->str() << std::endl;

		cout << "df/dx (10, -4) = " << f.derivative("x", 10, -4)->str() << std::endl;
	} catch (Barn <double, int> ::unknown_operation_overload_exception e) {
		cout << e.what() << endl;
	}

	try {
		Function <double, int> g = std::string("g(x, y) = x^2 + y = 54");
	} catch (Function <double, int> ::invalid_definition e) {
		cout << "Exception caught succesfully" << endl;
	}

	//////////////////////////////////////////////////////////////

	cout << endl << "==========================================================" << endl;
	cout << endl << "Node simplification...\n" << endl;

	Engine <double, int> eng("src/default_simplifications.ztp");

	node_manager<double, int> mg("x^2 * 1", {"x"});

	mg.print();

	mg.simplify(eng);

	//////////////////////////////////////////////////////////////

	cout << endl << "==========================================================" << endl;
	cout << endl << "Expression functions...\n" << endl;

	cout << "1: " << zhetapi::expr_str <double, int> ("2 * 4") << endl;

	int x = zhetapi::expr <int> ("3 * 6");

	cout << "x: " << x << endl;

	//////////////////////////////////////////////////////////////

	cout << endl << "==========================================================" << endl;
	cout << endl << "Tensor and indexing...\n" << endl;

	zhetapi::Tensor <double> tensor({4, 5, 3}, 4);

	cout << tensor << endl;

	tensor[{0, 0, 0}] = 45;

	cout << tensor << endl;

	zhetapi::Tensor <double> vector({3, 2}, {1, 2, 3, 4, 5, 6});

	cout << vector << endl;

	vector[{0, 1}] = 25;

	cout << vector << endl;
}