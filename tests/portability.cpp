// C/C++ headers
#include <iostream>
#include <vector>

// Engine headers
#include <function.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <tensor.hpp>

using namespace std;

// Tests
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

bool matrix_construction_and_memory()
{
	using namespace zhetapi;

	Matrix <double> tmp;

	cout << "Default constructor: " << tmp << endl;

	return true;
}

bool tensor_construction_and_memory()
{
	using namespace zhetapi;

	Tensor <double> tmp;

	cout << "Default: " << tmp << endl;

	return true;
}

// Testing rig
vector <pair <string, bool(*)()>> rig {
	{"vector construction and memory safety", &vector_construction_and_memory},
	{"matrix construction and memory safety", &matrix_construction_and_memory},
	{"tensor construction and memory safety", &tensor_construction_and_memory}
};

// Main program
int main()
{
	// Run tests in the test rig
	bool first = true;

	int count = 0;
	for (auto pr : rig) {
		if (first)
			first = false;
		else
			cout << endl;

		cout << "==============================" << endl;
		cout << "Running \"" << pr.first << "\" test:" << endl;

		cout << endl << "--------------------" << endl;
		bool tmp = pr.second();
		cout << "--------------------" << endl;

		if (tmp) {
			cout << endl << "\"" << pr.first << "\" test PASSED." << endl;
			count++;
		} else {
			cout << endl << "\"" << pr.first << "\" test FAILED." << endl;
		}

		cout << "==============================" << endl;
	}

	cout << endl << "Summary: passed " << count << "/" << rig.size() << " tests." << endl;
}
