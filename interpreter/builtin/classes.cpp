#include "builtin.hpp"

namespace zhetapi {

ZHETAPI_REGISTER(bt_dict)
{
	// No arguments
	return new Dictionary(TTargs {});
}

}
