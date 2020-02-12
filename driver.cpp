#include <iostream>

#include "src/variable.h"

using namespace std;

int main()
{
	variable <double> x("x", false, 12);

	cout << "Initial Value of Varaible: " << *x << endl;

	x[323.0];

	cout << "Value is Now: " << *x << endl;
}
