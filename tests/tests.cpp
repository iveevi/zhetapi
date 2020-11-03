// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <matrix.hpp>
#include <tensor.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
        Matrix <double> m {{3, 4}, {4, 6}};

        cout << m << endl;

        Matrix <double> mp = m;

        cout << mp << endl;
}
