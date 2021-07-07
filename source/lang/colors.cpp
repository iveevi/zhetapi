#include "../../engine/lang/colors.hpp"

namespace zhetapi {

namespace detail {

Warning warning;
Error error;

std::ostream &operator<<(std::ostream &os, const Warning &obj)
{
	os << "\033[1;33mWarning:\033[0m";
	return os;
}

std::ostream &operator<<(std::ostream &os, const Error &obj)
{
	os << "\033[1;31mError:\033[0m";
	return os;
}

}

}