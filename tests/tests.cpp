// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <vector.hpp>
#include <matrix.hpp>
#include <tensor.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
        Matrix <double> m1 {{3, 4}, {4, 6}};

        cout << m1 << endl;

        Matrix <double> m2 = m1;

        cout << m2 << endl;

	Matrix <double> m3(m2);

        cout << m3 << endl;

	Vector <double> v1 {1, 2, 3};

	cout << v1 << endl;

        Vector <double> v2 = v1;

        cout << v2 << endl;

	Vector <double> v3(v2);

        cout << v3 << endl;
}
