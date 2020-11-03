// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
// #include <matrix.hpp>
#include <tensor.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
        /* Matrix <double> m {{3, 4}, {4, 6}};

        cout << m << endl;

        Matrix <double> mp = m;

        cout << mp << endl; */

        Tensor <double> t1({2, 2}, {3, 4, 5, 6});

        cout << t1 << endl;

        Tensor <double> t2 = t1;

        cout << t2 << endl;
        
        Tensor <double> t3(t1);

        cout << t3 << endl;
}
