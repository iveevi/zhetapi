#ifndef ENODE_H_
#define ENODE_H_

// C/C++ headers
#include <cstdint>
#include <vector>
#include <exception>
#include <stdexcept>

// Engine headers
#include "primitive.hpp"
#include "primoptns.hpp"
#include "variant.hpp"

// TODO: separate source and header
namespace zhetapi {

// New node (expression node)
struct Enode {
	using Leaves = std::vector <Enode>;

	// Local type
	enum Type : uint8_t {
		etype_null,
		etype_operation,
		etype_primtive,
		etype_special,
		etype_miscellaneous
	};

	// Union saves space
	// - also allows us to greatly optimize prim-prim operations
	union Data {
		OpCode code;
		uint8_t misc;		// 1 for branching, 2 for while, ...
		Primitive *prim;
		Object *obj;		// representing special types
	} data;

	Type type; // 1 for op, 2 for prim, 3 for tok/spec type, 4 for misc
	std::vector <Enode> leaves;

	// Type specific constructors
	Enode();

	Enode(OpCode);
	Enode(OpCode, const Leaves &);
	Enode(OpCode, const Enode &, const Enode &);

	Enode(Primitive *);

	// Debugging stuff
	void print(int = 0, std::ostream & = std::cout) const;
};

// Printing
std::ostream &operator<<(std::ostream &, const Enode &);

// TODO: add symtab
Variant enode_value(const Enode &);

}

#endif
