#include "../include/autograd/autograd.hpp"
#include "../include/io/print.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	// TODO: test in testing dir
	// TODO: benchmark program
	Variable x, y;

	// Differentiation test
	// Function k = x/y;
	Function k = autograd::sqrt(x + x);
	Function h = k.differentiate(0);

	// TODO: input size exception
	std::cout << "h = " << h(2.0) << std::endl;
	std::cout << "h = " << h(4.0) << std::endl;
}
