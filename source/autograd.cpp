#include "../include/autograd/autograd.hpp"

namespace zhetapi {

namespace autograd {

Function operator+(const Variable &lhs, const Variable &rhs)
{
	return new_ <Add> ();
}

Function operator-(const Variable &lhs, const Variable &rhs)
{
	return new_ <Sub> ();
}

}

}
