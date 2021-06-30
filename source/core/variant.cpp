#include "../../engine/core/variant.hpp"

namespace zhetapi {

// Constructors
Variant::Variant()
		: data({.id = 0}) {}

Variant::Variant(const Primitive &prim)
		: data(Data {.prim = prim}) {}

Variant::Variant(const Object &obj)
		: data(Data {.obj = obj}) {}

// Variant type getter
size_t Variant::variant_type() const
{
	return (data.id) ? (1 + (data.id > MAX_PIDS)) : 0;
}

}