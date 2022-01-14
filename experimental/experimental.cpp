#include "../include/autograd/autograd.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	Constant a(1.0);
	Constant b(2.0);

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
	std::cout << "K: " << k(1.0, 2.0, 2.0) << std::endl;

	_sqrt *my_sqrt = new _sqrt();
	std::cout << "my_sqrt: " << my_sqrt->summary() << std::endl;

	_function *composed = my_sqrt->compose({f.get()});
	std::cout << "composed: " << composed->summary() << std::endl;

	std::cout << "composed: " << composed->compute({a, b}) << std::endl;

	std::cout << "sqrt: " << autograd::sqrt.summary() << std::endl;

	Function cfun = autograd::sqrt(f);

	std::cout << "cfun: " << cfun.summary() << std::endl;
}
