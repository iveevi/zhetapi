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

	Variable x, y;

	std::cout << "x = " << x.summary() << std::endl;
	std::cout << "y = " << y.summary() << std::endl;

	std::cout << autograd::pow.summary() << std::endl;

	// TODO: benchmark program
	Function f = autograd::sqrt(x); // autograd::pow(x, autograd::sqrt(y));

	std::cout << f.summary() << std::endl;
	std::cout << "f(4.5) = " << f(4.5) << std::endl;

	// NOTE: the first argument is y, not x
	Function g = autograd::pow(y * y + x, f);
	std::cout << g.summary() << std::endl;
	std::cout << "g(2.0, 3.0) = " << g(2.0, 3.0) << std::endl;
}
