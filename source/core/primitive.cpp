#include "../../engine/core/primitive.hpp"

namespace zhetapi {

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

Primitive p_bool(bool b)
{
	return {{.b = b}, id_bool};
}

Primitive p_int(long long int i)
{
	return {{.i = i}, id_int};
}

Primitive p_double(long double d)
{
	return {{.d = d}, id_double};
}

}
