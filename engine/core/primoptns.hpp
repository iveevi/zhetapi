#ifndef PRIMOPS_H_
#define PRIMOPS_H_

// C/C++ headers
#include <stdexcept>
#include <iostream>
#include <vector>

// Engine headers
#include "primitive.hpp"
#include "operations.hpp"

namespace zhetapi {

namespace core {

namespace primitive {

// operation type
using Operation = Primitive (*)(const Primitive &, const Primitive &);

struct Overload {
	OID		ovid;
	Operation	main;
};

// Opbase
using VOverload = std::vector <Overload>;

extern const VOverload operations[];

inline Primitive compute(OpCode code, const Primitive &a, const Primitive &b)
{
	// Static str tables (TODO: keep else where (str_tables...))
	static std::string id_strs[] {
		"null",
		"int",
		"double"
	};

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

	// Function (offset is l_add)
	const VOverload *ovb = &(operations[code]);

	for (OID i = 0; i < ovb->size(); i++) {
		OID ovid = (*ovb)[i].ovid;

		if (ovlid(a.id, b.id) == ovid)
			return (*ovb)[i].main(a, b);
	}

	// Throw here
	std::cout << "arg1.id = " << a.id << std::endl;
	std::cout << "arg2.id = " << b.id << std::endl;
	std::cout << "code = " << ovlid(a.id, b.id) << std::endl;
	throw bad_overload(code, a.id, b.id);

	return Primitive();
}

}

}

}

#endif
