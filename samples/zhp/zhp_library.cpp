// C++ headers
#include <vector>

// Engine include
#include <barn.hpp>

using namespace std;
using namespace zhetapi;

ZHETAPI_REGISTER(print_hello_world)
{
	cout << "Hello World!" << endl;

	return nullptr;
}
