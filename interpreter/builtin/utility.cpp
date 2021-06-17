#include "builtin.hpp"
#include <stdexcept>

namespace zhetapi {

// Integer range (checks for infinite-ness)
ZHETAPI_REGISTER(bt_range)
{
	Z start = 0;
	Z end = 0;
	Z stride = 1;

	OpZ *o1, *o2, *o3;
	switch (zhetapi_cast_cc(inputs, o1, o2, o3)) {
	case 1:
		end = o1->get();
		break;
	case 2:
		start = o1->get();
		end = o2->get();
		break;
	case 3:
		start = o1->get();
		end = o2->get();
		stride = o3->get();
		break;
	default:
		throw std::runtime_error("No matching overload for range");
	}

	// TODO: Check for infinite-ness
	Targs targs;
	for (Z i = start; i < end; i += stride)
		targs.push_back(new OpZ(i));

	return new Collection(targs);
}

}
