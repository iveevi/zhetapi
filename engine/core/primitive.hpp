#ifndef PRIMITIVE_H_
#define PRIMITIVE_H_

// C/C++ headers
#include <cstdint>
#include <string>

namespace zhetapi {

// Enode type IDs (at most 16) TODO: put inside primitive
enum TypeId : uint8_t {
	id_null,
	id_int,
	id_bool,
	id_double
};

// Primitive type
// either of bool, int or floating point (for now)
struct Primitive {
	// TODO: add the rest
	union {
		bool b;
		long long int i;
		long double d;
	} data;
	
	TypeId id;

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
