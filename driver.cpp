#include <iostream>
#include <vector>

#include "src/variable.h"
#include "src/stack.h"
#include "src/expression.h"
#include "src/var_stack.h"
// #include "src/func_stack.h"
// #include "src/application.h"
#include "src/functor.h"

using namespace std;

void test_variable()
{
	cout << endl << "BEGINNING VARIABLE TEST" << endl;

	variable <double> x("x", false, 12);

	cout << "Initial Value of Varaible: " << *x << endl;

	x[323.0];

	cout << "Value is Now: " << *x << endl;
}

void test_stack()
{
	cout << endl << "BEGINNING [SPLAY] STACK TEST" << endl;

	splay_stack <int> stack;

	cout << "Initial Stack: " << endl;
	stack.print();

	vector <int> vals {56, 1, 4, 8, 3, 9, 10};

	for (int i : vals)
		stack.insert(i);

	cout << "After Populating:" << endl;
	stack.print();

	cout << "Testing Find Function:" << endl;

	int temp;
	for (int i : vals) {
		cout << endl << "Trying to find " << i << endl;
		temp = stack.find(i);
		
		cout << "Returned " << temp << endl;
		stack.print();
	}
}

void test_expression()
{
	cout << endl << "BEGINNING EXPRESSION TEST" << endl;

	cout << "Expression evaluation with format: "
		<< expression <double> ::in_place_evaluate_formatted("12 * 13 * log 3 8")
		<< endl;

	string input;
	while (true) {
		cout << "Enter an expression to be evaluated: ";
		getline(cin, input);

		cout << "\t" << expression <double>
			::in_place_evaluate(input) << endl;
	}
}

void test_var_stack()
{
	var_stack <double> vst;

	vector <variable <double>> vals {variable <double> {"x", 12.0},
		variable <double> {"ran", true, 123},
		variable <double> {"y", 13423.423},
		variable <double> {"this", true, 12.0}};

	for (variable <double> v : vals) {
		cout << "\tInserting: " << v << endl;
		vst.insert(v);
	}

	cout << "After Populating:" << endl;
	vst.print();

	cout << "Testing Find Function:" << endl;

	variable <double> temp;
	for (variable <double> v : vals) {
		cout << endl << "Trying to find " << v << endl;
		temp = vst.find(v.symbol());
		
		cout << "Returned " << temp << endl;
		vst.print();
	}
}

void test_variable_parsing()
{
	var_stack <double> vst;

	vector <variable <double>> vals {variable <double> {"x", 12.0},
		variable <double> {"ran", true, 123},
		variable <double> {"y", 13423.423},
		variable <double> {"this", true, 12.0}};
	
	for (variable <double> v : vals) {
		// cout << "Inserting: " << v << endl;
		vst.insert(v);
	}

	cout << endl;

	string input = "6 * x + 5";
	cout << "Expression [" << input << "] = " << expression <double>
		::in_place_evaluate(input, vst) << endl;
	
	while (true) {
		cout << "Enter an expression to be evaluated: ";
		getline(cin, input);

		cout << "\t" << expression <double>
			::in_place_evaluate(input, vst) << endl;
	}
}

/* void test_application()
{
	application <double> app;
	var_stack <double> vst;

	vector <variable <double>> vals {variable <double> {"x", 12.0},
		variable <double> {"ran", true, 123},
		variable <double> {"y", 13423.423},
		variable <double> {"this", true, 12.0}};
	
	for (variable <double> v : vals) {
		// cout << "Inserting: " << v << endl;
		vst.insert(v);
	}

	string input;
	while (true) {
		cout << "Enter an expression to be evaluated: ";
		getline(cin, input);

		app = application <double> (input);
		app(vst);
	}
} */

void test_function()
{
	functor <double> f("f(x, y, z) = 3(sin x)(-5x^2 - y^2 + 3z)");
	// f.print();
	cout << "\t" << f({2, 3, 4}) << endl;
	cout << "\t" << f(2, 3, 4) << endl;
	cout << output(f) << endl;
}

/* void test_func_stack()
{
	func_stack <double> fst;

	vector <std::string> funcs {"f(x) = x^4 + 6",
		"h(x) = 232x^7 - 90", "g(x, y) = x^2 + y^2"};

	for (std::string str : funcs) {
		cout << "\tInserting: " << str << endl;
		fst.insert(functor <double> (str));
	}

	cout << "After Populating:" << endl;
	fst.print();

	cout << "Testing Find Function:" << endl;

	functor <double> temp("");
	for (std::string str : funcs) {
		cout << endl << "Trying to find " << str << endl;
		temp = fst.find(str.substr());
		
		//cout << "Returned " << temp << endl;
		//vst.print();
	}
} */

int main()
{
	test_function();
}
