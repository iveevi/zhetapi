#include "../../engine/core/enode.hpp"

namespace zhetapi {

// Enode
Enode::Enode() : data({.code = l_add}), type(etype_null) {}

Enode::Enode(Primitive *prim) : data(Data {.prim = prim}), type(etype_null) {}

Enode::Enode(Object *obj) : data(Data {.obj = obj}), type(etype_null) {}

Enode::Enode(OpCode code, const Leaves &lvs)
		: data(Data {.code = code}), type(etype_operation),
		leaves(lvs) {}

Enode::Enode(OpCode code, const Enode &e1, const Enode &e2)
		: data(Data {.code = code}), type(etype_operation),
		leaves({e1, e2}) {}

void Enode::print(int indent, std::ostream &os) const
{
	// Static value -> string tables
	static std::string type_strs[] {
		"operation",
		"primitive",
		"special",
		"miscellaneous"
	};

	static std::string op_strs[] {
		"addition",
		"subtraction",
		"multiplication",
		"division"
	};

	std::string indstr(indent, '\t');

	os << indstr << "[" << type_strs[type] << "] ";

	std::string main;

	switch (type) {
	case etype_operation:
		main = op_strs[data.code];
		break;
	case etype_primtive:
		main = data.prim->str();
		break;
	case etype_object:
	case etype_miscellaneous:
	default:
		main = "?";
		break;
	}

	os << main << std::endl;
	for (const Enode &en : leaves)
		en.print(indent + 1, os);
}

std::ostream &operator<<(std::ostream &os, const Enode &en)
{
	en.print(0, os);

	return os;
}

Variant op_prim_prim(OpCode code, const Variant &v1, const Variant &v2)
{
	// TODO: add cast functions (returning objects, not pointers)
	return new Primitive(do_prim_optn(code, *((Primitive *) v1), *((Primitive *) v2)));
}

Variant enode_value(const Enode &en)
{
	Variant v1;
	Variant v2;

	Primitive *p1 = nullptr;
	Primitive *p2 = nullptr;

	uint8_t sum;

	switch (en.type) {
	case Enode::etype_operation:
		// Need at least one arg
		v1 = enode_value(en.leaves[0]);
		if (en.leaves.size() == 2)
			v2 = enode_value(en.leaves[1]);

		// TODO: use data.id later
		sum = variant_type(v1) + (variant_type(v2) << 32);
		if (sum == 5) { // prim & prim
			return op_prim_prim(en.data.code, v1, v2);
		} else {
			throw std::runtime_error("null (or invalid) first op");
		}
		
		break;
	case Enode::etype_primtive:
		return Variant(&en.data.prim);
	case Enode::etype_object:
		return Variant(&en.data.obj);
	case Enode::etype_miscellaneous:
		break;
	default:
		break;
	}

	throw std::runtime_error("enode_value err");
}

}
