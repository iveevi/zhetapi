#include "../include/autograd/autograd.hpp"
#include "../include/autograd/ml.hpp"
#include "../include/io/print.hpp"
#include "../include/common.hpp"
// #include "../include/common.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	// MSE function (5 inputs)
	Variable x;
	Variable y;

	auto mse = square(norm(x - y))/Constant(5);

	std::cout << "mse:\n" << mse.summary() << std::endl;
	std::cout << "mse(1, 0): " << mse(2, 0) << std::endl;

	auto dmse = mse.differentiate(0);
	std::cout << "\ndmse:\n" << dmse.summary() << std::endl;
	std::cout << "dmse(1, 0): " << dmse(2, 0) << std::endl;

	// auto dense1 = ml::dense(5, 5);
	ml::_kdense dense1(5, 5);

	Constant c {1.0, 2.0, 3.0, 4.0, 5.0};
	std::cout << "\nc = " << c << std::endl;

	// std::cout << "dense1(c) = " << dense1(c) << std::endl;
	std::cout << "dense1(c) = " << dense1.compute({c}) << std::endl;
	std::cout << dense1.summary() << std::endl;

	// Gradient
	// std::cout << "gradient: " << dense1.gradient(c) << std::endl;
	Constant igrad {{5, 1}, 1};
	std::cout << "gradient: " << dense1.gradient(igrad, {c}) << std::endl;
}
