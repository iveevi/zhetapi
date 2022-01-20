#include "global.hpp"

TEST(vector_construction_and_memory)
{
	using namespace zhetapi;

	Vector <double> tmp;

	oss << "Default constructor: " << tmp << endl;

	tmp = Vector <double> {1, 1, 1, 4, 6};

	oss << "Initializer list constructor: " << tmp << endl;

	vector <double> nums {1, 6, 3, 8, 1};

	tmp = Vector <double> (nums);

	oss << "Vector construction and assignment operator: " << tmp << endl;

	tmp = Vector <double> (4, 6);

	oss << "Size and default value constructor: " << tmp << endl;

	/* double *arr = new double[10];
	for (size_t i = 0; i < 10; i++)
		arr[i] = pow(i, 3.14);

	tmp = Vector <double> (10, arr);

	oss << "Size and pointer constructor: " << tmp << endl;

	tmp = Vector <double> (5, arr);

	oss << "Cropped size and pointer constructor: " << tmp << endl;

	// Free resources
	delete[] arr; */

	return true;
}

TEST(vector_operations)
{
	using namespace zhetapi;

	Vector <int> a = {1, 2, 3};
	Vector <int> b = {4, 5, 6};
	Vector <int> c = {1, 2, 3, 4, 5, 6};

	Vector <int> cc = {
		1, 2, 3, 4, 5, 6,
		1, 2, 3, 4, 5, 6,
	};

	oss << "a = " << a << endl;
	oss << "b = " << b << endl;

	oss << concat(a, b) << endl;
	oss << c << endl;

	if (c != concat(a, b)) {
		oss << "Concatenation is incorrect." << endl;

		return false;
	} else {
		oss << "Concatenation passed." << endl;
	}

	oss << "\na o b = " << concat(a, b) << endl;
	oss << "c = " << c << endl;

	oss << cc << endl;
	oss << concat(a, b, c) << endl;

	if (cc != concat(a, b, c)) {
		oss << "(Variadic) Concatenation is incorrect." << endl;

		return false;
	} else {
		oss << "(Variadic) Concatenation passed." << endl;
	}

	return true;
}
