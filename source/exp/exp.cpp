#include <function.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Function f;
	
	f = "f(x) = x^2";

	f.print();

	f = "f(x) = (x^2) * (dx)";

	f.print();
}
