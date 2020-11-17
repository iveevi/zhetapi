#include "port.hpp"

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
