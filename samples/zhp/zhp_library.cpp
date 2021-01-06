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

extern "C" {

ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT(print_hello_world);
	ZHETAPI_EXPORT_SYMBOL(hello, print_hello_world);
}

}
