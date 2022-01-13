#include "../include/autograd/autograd.hpp"

using namespace zhetapi::autograd;

Function add(Variable a, Variable b)
{
	return a + b;
}

int main()
{
	Constant a = Constant(1.0);
	Constant b = Constant(2.0);

	std::cout << a << std::endl;
	std::cout << b << std::endl;

	Function f = add({}, {});
	std::cout << f(a, b) << std::endl;

	Variable x, y, z;
	Function g = x + y;

	std::cout << g(a, b) << std::endl;
}
