#include <equation.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Equation eq({"x^2 + x", "1"});

	cout << "EQ: " << eq << endl;
	cout << "#args = " << eq.args() << endl;

	eq.solve();
}
