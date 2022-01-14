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

	std::cout << autograd::exp.summary() << std::endl;
	std::cout << autograd::sqrt.summary() << std::endl;

	std::cout << autograd::exp(10) << std::endl;

	// Function f = x * x + a * x + b;
	Function f = autograd::sqrt(autograd::exp);

	std::cout << f.summary() << std::endl;
	std::cout << "f(2.0) = " << f(2.0) << std::endl;

	std::cout << io::table(
		{"x", "f(x)", "error"}, {
			{ "2.0", "f(2.0)", "0.0" },
			{ "3.0", "f(3.0)", "0.0" },
			{ "4.0", "f(4.0)", "0.0" },
			{ "5.0", "f(5.0)", "0.0" },
			{ "6.0", "f(6.0)", "0.0" },
			{ "7.0", "f(7.0)", "0.0" },
			{ "8.0", "f(8.0)", "0.0" },
			{ "9.0", "f(9.0)", "0.0" },
			{ "10.0", "f(10.0)", "0.0" },
			{ "11.0", "f(11.0)", "0.0" },
			{ "12.0", "f(12.0)", "0.0" },
		}) << std::endl;
}
