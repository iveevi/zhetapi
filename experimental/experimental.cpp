#include "../include/autograd/autograd.hpp"
#include "../include/autograd/ml.hpp"
#include "../include/autograd//activation.hpp"
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
	auto true_dmse = Constant(2) * (x - y)/Constant(5);
	auto dmse = mse.differentiate(0);

	// Target value
	Constant target {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	Constant input {2.0, 2.0, 2.0, 2.0, 2.0};

	// Dense layer
	ISeq model;

	model.append(
		new Get(0),
		new ml::_kdense(5, 7),
		new ml::_relu::kernel(),
		new ml::_kdense(7, 7),
		new ml::_leaky_relu::kernel(0.1)
	);

	std::cout << "\nSummary:\n" << model.summary() << std::endl;

	auto output = model.compute({input});
	output.flatten();

	std::cout << "Out: " << output << std::endl;
	std::cout << "\terror = " << mse(output, target) << std::endl;
	std::cout << "\tderror = " << dmse(output, target) << std::endl;
	std::cout << "\ttrue = " << true_dmse(output, target) << std::endl;

	std::cout << "\nStarting training loop:" << std::endl;

	// Training loop
	for (int i = 0; i < 10000; i++) {
		auto output = model.compute({input}).flat();
		auto igrad = true_dmse(output, target);
		auto grads = model.gradient({igrad});

		// TOOD: optimizers
		for (auto &J : grads.grads)
			J *= -0.01;

		std::cout << "\nOutput after " << i << " iterations:\n" << output << std::endl;
		std::cout << "\tDMSE = " << igrad << std::endl;
		std::cout << "\tigrads = " << grads.igrads << std::endl;
		std::cout << "\nGrads after " << i << " iterations:\n" << grads.grads << std::endl;

		model.update_parameters(grads.grads);
	}

	/* Constant x1 {0, -1, 2, 3, -5, -6, 7};
	auto f = ml::leaky_relu(0.1);
	for (int i = 0; i < 10; i++) {
		auto output = f(x1);
		auto igrad = true_dmse(output, target);
		auto grads = f.gradient({igrad});

		std::cout << "\nOutput after " << i << " iterations:\n" << output << std::endl;
		std::cout << "\tDMSE = " << igrad << std::endl;
		std::cout << "\tigrads = " << grads.igrads << std::endl;

		x1 += -0.01l * grads.igrads[0];
	} */
}
