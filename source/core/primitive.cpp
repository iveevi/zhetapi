#include "../../engine/core/primitive.hpp"

namespace zhetapi {

Primitive::Primitive() : data({.b = false}), id(id_null) {}
Primitive::Primitive(bool b) : data({.b = b}), id(id_bool) {}
Primitive::Primitive(long long int i) : data({.i = i}), id(id_int) {}
Primitive::Primitive(long double d) : data({.d = d}), id(id_double) {}

std::string Primitive::str() const
{
	switch (id) {
	case id_bool:
		return (data.b) ? "true" : "false";
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
