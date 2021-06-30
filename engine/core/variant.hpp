#ifndef VARIANT_H_
#define VARIANT_H_

// Engine headers
#include "primitive.hpp"
#include "object.hpp"

namespace zhetapi {

// Value type (special type or primitive) TODO: switch to variant
struct Variant {
	// id 0 = null, id <= MAX_PID = prim, id > MAX_PID = variant
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
};

}

#endif