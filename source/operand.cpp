#include <operand.hpp>

namespace zhetapi {

// Specializations
template <>
std::string Operand <bool> ::dbg_str() const
{
	return (_val) ? "true" : "false";
}

}