#include <iostream>
#include <vector>

#include "src/variable.h"
#include "src/stack.h"
#include "src/expression.h"

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

	string input;
	while (true) {
		cout << "Enter an expression to be evaluated: ";
		getline(cin, input);

		cout << "\t" << expression <double>
			::in_place_evaluate(input) << endl;
	}
}

int main()
{
	test_expression();
}
