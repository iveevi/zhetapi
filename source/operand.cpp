#include <operand.hpp>

namespace zhetapi {

// Specializations
template <>
std::string Operand <bool> ::str() const
{
	return (_val) ? "true" : "false";
}

}