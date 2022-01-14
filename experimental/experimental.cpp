#include "../include/autograd/autograd.hpp"

using namespace zhetapi::autograd;

int main()
{
	Constant a = Constant(1.0);
	Constant b = Constant(2.0);

	std::cout << a << std::endl;
	std::cout << b << std::endl;

	Variable x, y, z;
	Function f = x + y;
	Function g = x + z;

	std::cout << "G: " << g(a, b) << std::endl;

	// std::cout << f.summary() << std::endl;
	// std::cout << g.summary() << std::endl;

	Function h = f + g;
	// std::cout << h.summary() << std::endl;
	std::cout << "H: " << h(a, b, b) << std::endl;

	Function k = f * z;
	std::cout << k.summary() << std::endl;
	std::cout << "K: " << k(a, b, b) << std::endl;

	// TODO: benchmark program
}
