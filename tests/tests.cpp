// C/C++ headers
#include <iostream>

// Engine headers
#include <matrix.hpp>
#include <vector.hpp>
#include <tensor.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Tensor <double> t({3, 4, 2}, 5);

	cout << t << endl;

	t[{0, 0, 0}] = 56;

	cout << t << endl;
}