#ifndef GENOPTNS_H_
#define GENOPTNS_H_

// Standard headers
#include <iostream>

// Engine headers
#include "operations.hpp"
#include "variant.hpp"
#include "primoptns.hpp"

namespace zhetapi {

namespace core {

// operation type
using Operation = Variant (*)(const Variant, const Variant);

struct Overload {
	OID		ovid;
	Operation	main;
};

// Opbase
using VOverload = std::vector <Overload>;

extern const std::vector <VOverload> operations;

inline Variant compute(OpCode code, const Variant a, const Variant b)
{
	// Check if a and b are primitives
	if (is_primitive(a) && is_primitive(b))
		return new Primitive(primitive::compute(code, *((Primitive *) a), *((Primitive *) b)));

	// Static str tables (TODO: keep else where (str_tables...))
	static std::string id_strs[] {
		"null",
		"int",
		"double"
	};

	// TODO: static str tables
	static std::string op_strs[] {
		"get",
		"const",
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

	// Check bounds of code
	if (code > operations.size())
		throw std::runtime_error("code is out of bounds");

	// Function (offset is l_add)
	const VOverload *ovb = &(operations[code]);

	// Get ids of the operands
	TypeId aid = gid(a);
	TypeId bid = gid(b);

	for (uint8_t i = 0; i < ovb->size(); i++) {
		uint64_t ovid = (*ovb)[i].ovid;

		if (ovid == ovlid(aid, bid))
			return (*ovb)[i].main(a, b);
	}

	// Throw here
	std::cout << "arg1.id = " << aid << std::endl;
	std::cout << "arg2.id = " << bid << std::endl;
	throw bad_overload(code, aid, bid);

	return Variant();
}

}

}

#endif