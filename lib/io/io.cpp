// C++ headers
#include <vector>

// Engine include
#include <barn.hpp>

using namespace std;
using namespace zhetapi;

ZHETAPI_REGISTER(__zhp_std_printf)
{
	cout << "Printf!" << endl;

	return nullptr;
}

ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT_SYMBOL(printf, __zhp_std_printf);
}
