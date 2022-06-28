#include "../include/autograd/autograd.hpp"
#include "../include/autograd/ml.hpp"
#include "../include/autograd/activation.hpp"
#include "../include/autograd/optimizer.hpp"
#include "../include/io/print.hpp"
#include "../include/common.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	// MSE function (5 inputs)
	Variable x;
	Variable y;

	auto mse = square(norm(x - y))/Constant(5);
	auto true_dmse = Constant(2) * (x - y)/Constant(5);
	auto dmse = mse.differentiate(0);

	// Target value
	Constant target {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	Constant input {2.0, 2.0, 2.0, 2.0, 2.0};

	// Function API
	auto model = ml::dense(5, 7)(x);
	model = ml::tanh(model);
	model = ml::dense(7, 7)(model);
	model = ml::leaky_relu(0.1)(model);

	std::cout << "\nSummary:\n" << model.summary() << std::endl;
	std::cout << "# of parameters = " << model.parameters() << std::endl;
	std::cout << "# of tunable parameters = " << model.tunable_parameters() << std::endl;

	auto output = model(input);
	output.flatten();

	// Optimizer
	auto optimizer = ml::Adam(model.parameters(), 0.01);

	std::cout << "\nOut: " << output << std::endl;
	std::cout << "\terror = " << mse(output, target) << std::endl;
	std::cout << "\tderror = " << dmse(output, target) << std::endl;
	std::cout << "\ttrue = " << true_dmse(output, target) << std::endl;

	std::cout << "\nStarting training loop:" << std::endl;

	// Training loop
	for (int i = 0; i < 100; i++) {
		auto output = model(input).flat();
		auto igrad = true_dmse(output, target);
		auto grads = model.gradient({igrad});

		/* TOOD: optimizers
		for (auto &J : grads.grads)
			J *= -0.01; */

		optimizer(grads.grads);

		std::cout << "\nOutput after " << i << " iterations:\n" << output << std::endl;
		std::cout << "\tDMSE = " << igrad << std::endl;
		/* std::cout << "\tigrads = " << grads.igrads << std::endl;
		std::cout << "\nGrads after " << i << " iterations:\n" << grads.grads << std::endl; */

		model.update_parameters(grads.grads);
	}
}
