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
	Variable x, y;

	auto model = ml::dense(3, 3);
	model = ml::sigmoid(model);
	// model = ml::dense(3, 3)(model);
	// model = ml::sigmoid(model);

	Constant input {1.0f, 1.0f, 1.0f};
	Constant target {0.3f, 0.5f, 0.7f};

	// TODO: make a standard function for MSE, etc
	auto loss = square(length(x - y))/Constant(3);
	auto dloss = 2 * (x - y) / Constant(3);

	auto output = model(input);
	std::cout << "Output: " << output << std::endl;

	auto error = loss(output.flat(), target);
	std::cout << "Error: " << error << std::endl;

	auto igrad = dloss(output.flat(), target);
	std::cout << "Gradient: " << igrad << std::endl;

	auto grads = model.gradient({igrad});
	std::cout << "Gradients:" << std::endl;
	for (auto &g : grads.grads)
		std::cout << g << std::endl;
}
