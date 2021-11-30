#ifndef VARIANT_H_
#define VARIANT_H_

// Engine headers
#include "primitive.hpp"
#include "object.hpp"

namespace zhetapi {

// Use pointer style casting to save memory
using Variant = void *;

// id 0 = null, id <= MAX_PID = prim, id > MAX_PID = variant
Variant null_variant();

TypeId variant_type(const Variant);
std::string variant_str(const Variant);

// Get the id of variant
constexpr TypeId gid(const Variant var)
{
	return *((TypeId *) var);
}

// Check primitive type
constexpr bool is_primitive(const Variant var)
{
	return gid(var) <= MAX_PRIMITIVE_IDS;
}

}

#endif
