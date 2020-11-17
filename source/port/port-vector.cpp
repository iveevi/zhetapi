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

	delete arr;

	return true;
}