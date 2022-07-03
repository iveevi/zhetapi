#include "../include/autograd/activation.hpp"
#include "../include/autograd/autograd.hpp"
#include "../include/autograd/ml.hpp"
#include "../include/autograd/optimizer.hpp"
#include "../include/autograd/train.hpp"
#include "../include/common.hpp"
#include "../include/io/print.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	/* Variable x, y;

	auto model = ml::dense(3, 5)(x);
	model = ml::sigmoid(model);
	model = ml::dense(5, 5)(model);
	model = ml::softmax(model);

	auto optimizer = ml::RMSprop(model.parameters(), 0.01);

	Constant input {1.0f, 1.0f, 1.0f};
	Constant target {0.3f, 0.3f, 0.2f, 0.2f, 0.2f};

	// TODO: make a standard function for MSE, etc
	auto loss = square(length(x - y))/Constant(5);
	auto dloss = 2 * (x - y) / Constant(5);

	for (size_t n = 0; n < 3; n++) {
		auto output = model(input);

		std::cout << "\n=========================\n";
		std::cout << "Output: " << output << std::endl;

		auto error = loss(output.flat(), target);
		std::cout << "Error: " << error << std::endl;

		auto igrad = dloss(output.flat(), target);
		std::cout << "Gradient: " << igrad << std::endl;

		auto grads = model.gradient({input}, {igrad});
		optimizer(grads.grads);

		model.update_parameters(grads.grads);
	} */

	auto optimizer = ml::Adam(2, 0.01);
	
	auto gq = GradientQueue {
		Constant {1.0f, 1.0f, 1.0f},
		Constant {0.3f, 0.3f, 0.2f, 0.2f, 0.2f}
	};

	std::cout << "gq = " << gq << std::endl;

	optimizer(gq);
	std::cout << "\ngq = " << gq << std::endl;
	
	optimizer(gq);
	std::cout << "\ngq = " << gq << std::endl;
}
