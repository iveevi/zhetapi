#include "../../engine/core/primitive.hpp"

namespace zhetapi {

Primitive::Primitive() : id(id_null) {}
Primitive::Primitive(long long int i) : data({.i = i}), id(id_int) {}
Primitive::Primitive(long double d) : data({.d = d}), id(id_double) {}

std::string Primitive::str() const
{
	switch (id) {
	case id_int:
		return std::to_string(data.i);
	case id_double:
		return std::to_string(data.d);
	default:
		break;
	}

	return "?";
}

}
