#include "../include/autograd/autograd.hpp"
#include "../include/io/print.hpp"
// #include "../include/common.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	// TODO: test in testing dir
	// TODO: benchmark program
	Variable x, y;

	// Differentiation test
	/* Function k = x/y;
	Function k = autograd::sqrt(x + x);
	Function h = k.differentiate(0);

	// TODO: input size exception
	std::cout << "k = " << k(2.0) << ", h = " << h(2.0) << std::endl;
	std::cout << "k = " << k(4.0) << ", h = " << h(4.0) << std::endl; */

	Constant shape {2.0, 2.0};
	std::cout << "shape = " << shape << std::endl;

	Function g = autograd::reshape(shape, y);
	std::cout << "g:" << std::endl;
	std::cout << g.summary() << std::endl;

	// TODO: initialier list constructor
	// Constant c = {2.0, 2.0, 2.0, 2.0};
	
	Constant c(Constant::shape_type {4}, 2.0);

	std::cout << "c = " << c << std::endl;
	std::cout << "g(c) = " << g(c) << std::endl;
}
