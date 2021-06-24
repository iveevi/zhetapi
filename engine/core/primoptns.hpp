#ifndef PRIMOPS_H_
#define PRIMOPS_H_

// C/C++ headers
#include <stdexcept>
#include <iostream>
#include <vector>

// Engine headers
#include "primitive.hpp"

namespace zhetapi {

enum OpCode : uint8_t {
	l_add,
	l_sub,
	l_mul,
	l_div
};

// 8 bits is sufficient
using OvlId = uint8_t;

constexpr OvlId ovlid(TypeId a, TypeId b)
{
	return a + (b << 4);
}

// operation type
using optn = Primitive (*)(const Primitive &, const Primitive &);

struct ovl {
	uint8_t ovid;
	optn main;
};

// Opbase
using ovlbase = std::vector <ovl>;

extern const ovlbase opbase[];

inline Primitive do_prim_optn(OpCode code, const Primitive &arg1, const Primitive &arg2)
{
	// Static str tables (TODO: keep else where (str_tables...))
	static std::string id_strs[] {
		"null",
		"bool",
		"int",
		"double"
	};
	
	static std::string op_strs[] {
		"addition",
		"subtraction",
		"multiplication",
		"division"
	};

	// Exception
	class bad_overload : public std::runtime_error {
	public:
		// TODO: fill out the rest of the error
		bad_overload(OpCode code, TypeId a, TypeId b) : std::runtime_error("Bad overload ("
				+ id_strs[a] + ", " + id_strs[b] + ") for operation \""
				+ op_strs[code] + "\"") {}
	};

	// Function
	const ovlbase *ovb = &(opbase[code]);

	for (uint8_t i = 0; i < ovb->size(); i++) {
		uint8_t ovid = (*ovb)[i].ovid;

		if (((ovid & 0x0F) == arg1.id) && (((ovid & 0xF0) >> 4) == arg2.id))
			return (*ovb)[i].main(arg1, arg2);
	}

	// Throw here
	throw bad_overload(code, arg1.id, arg2.id);

	return Primitive();
}

}

#endif
