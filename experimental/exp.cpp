#include <complex.hpp>
#include <function.hpp>

using namespace zhetapi;
using namespace std;

int main()
{
	Function f = "f(x) = x + x";

	f.print();

	cout << f << endl;
}
