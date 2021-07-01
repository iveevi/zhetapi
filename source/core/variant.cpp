#include "../../engine/core/variant.hpp"

namespace zhetapi {

Variant null_variant()
{
	// Id needs to equal 0
	return new size_t(0);
}

// Variant type getter
TypeId variant_type(Variant var)
{
	TypeId id = *((TypeId *) var);
	return (id) ? (1 + (id > MAX_PIDS)) : 0;
}

std::string variant_str(Variant var)
{
	switch (variant_type(var)) {
	case 0:
		return "<Null>";
	case 1:
		return ((Primitive *) var)->str();
	case 2:
	default:
		break;
	}

	return "?";
}

}