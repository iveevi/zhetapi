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
	Function k = x/y;
	Function h = k.differentiate(0);

	std::cout << "h = " << h(2.0, 4.0) << std::endl;
	std::cout << "h = " << h(4.0, 2.0) << std::endl;
}
