#include "../include/autograd/ml.hpp"
#include "../include/autograd/activation.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

#include <unistd.h>

int main()
{
	{
		Variable x;

		auto model = ml::dense(10, 5)(x);
		model = ml::sigmoid(model);
		model = ml::dense(5, 3)(model);
		model = ml::softmax(model);

		std::cout << "Model:\n" << model.summary() << std::endl;

		GradientQueue gq;

		Constant in {Constant::shape_type {10}, 0.1};

		std::cout << "in = " << in << std::endl;
		std::cout << "out = " << model(in) << std::endl;

		Constant igrad {Constant::shape_type {3}, 1.0};

		std::cout << "igrad = " << igrad << std::endl;

		Gradient grads = model.gradient({in}, {igrad});

		std::cout << "grads:" << std::endl;
		for (auto &p : grads.grads)
			std::cout << "\t" << p << std::endl;

		detail::MemoryTracker::report();
	}

	detail::MemoryTracker::report();
}
