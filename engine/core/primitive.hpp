#ifndef PRIMITIVE_H_
#define PRIMITIVE_H_

// C/C++ headers
#include <cstdint>
#include <string>

namespace zhetapi {

// Maximum ID of a primitive
#define MAX_PRIMITIVE_IDS	3

// ID type alias
using TypeId = uint32_t;

// Enode type IDs (at most 16) TODO: put inside primitive
enum PrimIds : TypeId {
	id_null,
	id_bool,
	id_int,
	id_double
};

// Primitive type
// either of bool, int or floating point (for now)
struct Primitive {
	// ID header
	TypeId id;

	// TODO: add the rest
	union {
		bool b;
		long long int i;
		long double d;
	} data;

	// Constructors
	Primitive();
	Primitive(bool);
	Primitive(long long int);
	Primitive(long double);

	// Methods
	std::string str() const;
};

}

#endif
