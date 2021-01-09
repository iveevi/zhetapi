// C++ headers
#include <vector>

// Engine include
#include <zhplib.hpp>

using namespace std;
using namespace zhetapi;

ZHETAPI_REGISTER(print_hello_world)
{
	cout << "Hello World!" << endl;

	return nullptr;
}

ZHETAPI_REGISTER(get_first)
{
	return (inputs.size() > 0) ? inputs[0] : nullptr;
}

ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT(print_hello_world);
	ZHETAPI_EXPORT_SYMBOL(hello, print_hello_world);
	ZHETAPI_EXPORT_SYMBOL(first, get_first);
}
