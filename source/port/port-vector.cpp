#include "port.hpp"

bool vector_construction_and_memory()
{
	using namespace zhetapi;

	Vector <double> tmp;

	cout << "Default constructor: " << tmp << endl;

	tmp = Vector <double> {1, 1, 1, 4, 6};

	cout << "Initializer list constructor: " << tmp << endl;

	vector <double> nums {1, 6, 3, 8, 1};

	tmp = Vector <double> (nums);

	cout << "Vector construction and assignment operator: " << tmp << endl;

	tmp = Vector <double> (4, 6);

	cout << "Size and default value constructor: " << tmp << endl;

	double *arr = new double[10];
	for (size_t i = 0; i < 10; i++)
		arr[i] = pow(i, 3.14);
	
	tmp = Vector <double> (10, arr);

	cout << "Size and pointer constructor: " << tmp << endl;
	
	tmp = Vector <double> (5, arr);
	
	cout << "Cropped size and pointer constructor: " << tmp << endl;

	return true;
}

bool vector_operations()
{
	using namespace zhetapi;

	Vector <int> a = {1, 2, 3};
	Vector <int> b = {4, 5, 6};
	Vector <int> c = {1, 2, 3, 4, 5, 6};

	cout << "a = " << a << endl;
	cout << "b = " << b << endl;

	cout << concat(a, b) << endl;
	cout << c << endl;

	if (c != concat(a, b)) {
		cout << "Concatenation is incorrect." << endl;

		return false;
	} else {
		cout << "Concatenation passed." << endl;
	}

	if (concat(c, c) != concat(a, b, c)) {
		cout << "(Variadic) Concatenation is incorrect." << endl;

		return false;
	} else {
		cout << "(Variadic) Concatenation passed." << endl;
	}

	return true;
}
