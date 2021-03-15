#include <equation.hpp>

using namespace zhetapi;
using namespace std;

int main()
{
	Equation eq({
		"sin(x)^2 + cos(x)^2",
		"1"
	});

	cout << eq.representation() << endl;
}
