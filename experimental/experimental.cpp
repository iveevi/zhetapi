#include "../include/autograd/autograd.hpp"
#include "../include/io/print.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	Constant a(1.0);
	Constant b(2.0);

	std::cout << a << std::endl;
	std::cout << b << std::endl;

	Variable x;

	// TODO: benchmark program
	Function f = autograd::pow(x, x);
	Function pf1 = autograd::pow(2.0, x);
	Function pf2 = autograd::pow(x, 2.0);
	std::cout << "--> " << autograd::pow(2.0, 5.0) << std::endl;
	std::cout << "--> " << pf1(5.0) << std::endl;
	std::cout << "--> " << pf2(2.0) << std::endl;

	/* std::cout << autograd::pow.summary() << std::endl;
	std::cout << f.summary() << std::endl;
	std::cout << pf1.summary() << std::endl;
	std::cout << pf2.summary() << std::endl; */
}
