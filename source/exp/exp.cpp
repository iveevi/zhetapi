#include <std/interval.hpp>

using namespace std;
using namespace zhetapi::utility;

int main()
{
	Interval <> a(1, 5);
	Interval <> b(6, 7, false);

	cout << "a: " << a << endl;
	cout << "b: " << b << endl;

	cout << "a | b: " << (a | b) << endl;

	cout << "length: " << (a | b).size() << endl;
}
