#include <operand.hpp>

#include <core/types.hpp>

// Macros
#define id_spec(type)					\
	template <>					\
	uint8_t Operand <type> ::id() const		\
	{						\
		return zhp_id <Operand <type>> ();	\
	}

namespace zhetapi {

// Specializations
template <>
std::string Operand <bool> ::dbg_str() const
{
	return (_val) ? "true" : "false";
}

template <>
std::string Operand <Args> ::dbg_str() const
{
	std::string out = "<Args> {";

	for (size_t i = 0; i < _val.size(); i++) {
		out += _val[i];

		if (i < _val.size() - 1)
			out += ", ";
	}

	return out + "}";
}

template <>
std::string Operand <Vector <Token *>> ::dbg_str() const
{
	std::string out = "[";

	// TODO: need to implement a specialized deconstructor for
	// Vector <Token *> (and maybe other vector of pointer types)
	for (size_t i = 0; i < _val.size(); i++) {
		out += _val[i]->dbg_str();

		if (i < _val.size() - 1)
			out += ", ";
	}

	return out + "]";
}

template <>
std::string Operand <Matrix <Token *>> ::dbg_str() const
{
	std::string out = "[";

	// TODO: need to implement a specialized deconstructor for
	// Matrix <Token *> (and maybe other matrix of pointer types)
	for (size_t i = 0; i < _val.get_rows(); i++) {
		out += "[";

		for (size_t j = 0; j < _val.get_cols(); j++) {
			out += _val[i][j]->dbg_str();

			if (j < _val.get_cols() - 1)
				out += ", ";
		}

		out += "]";
		if (i < _val.get_rows() - 1)
			out += ", ";
	}

	return out + "]";
}

// ID specs
id_spec(Z)
id_spec(Q)
id_spec(R)

id_spec(B)
id_spec(S)

id_spec(CmpZ)
id_spec(CmpQ)
id_spec(CmpR)

id_spec(VecZ)
id_spec(VecQ)
id_spec(VecR)

id_spec(VecCmpZ)
id_spec(VecCmpQ)
id_spec(VecCmpR)

id_spec(MatZ)
id_spec(MatQ)
id_spec(MatR)

id_spec(MatCmpZ)
id_spec(MatCmpQ)
id_spec(MatCmpR)

}
