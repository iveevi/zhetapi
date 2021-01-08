#include "l.hpp"

ZHETAPI_REGISTER(print_hello_world)
{
	cout << "Hello World!" << endl;

	return nullptr;
}

ZHETAPI_REGISTER(get_first)
{
	return (inputs.size() > 0) ? inputs[0] : nullptr;
}
