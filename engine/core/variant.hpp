#ifndef VARIANT_H_
#define VARIANT_H_

// Engine headers
#include "primitive.hpp"
#include "object.hpp"

namespace zhetapi {

// Use pointer style casting to save memory
using Variant = void *;

/* Value type (special type or primitive) TODO: switch to variant
struct Variant {
	union Data {
		TypeId		id;
		Primitive	prim;
		Object		obj;
	} data;

	// Constructors
	Variant();			// Null variant
	Variant(const Primitive &);
	Variant(const Object &);

	// Variant type
	size_t variant_type() const;

	// Methods
	std::string str() const;
}; */

// id 0 = null, id <= MAX_PID = prim, id > MAX_PID = variant
Variant null_variant();

TypeId variant_type(const Variant);
std::string variant_str(const Variant);

// TODO: is_primitive function (inline)

}

#endif
