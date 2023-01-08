#include "../include/autograd/ml.hpp"
#include "../include/autograd/activation.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	{
		Variable x;

		auto model = ml::dense(4, 3, "debug")(x);
		model = ml::sigmoid(model);
		model = ml::dense(3, 2, "debug")(model);
		model = ml::sigmoid(model);
		
		std::cout << "Summary:" << std::endl;
		std::cout << model.summary() << std::endl;

		Constant in {1.0f, 2.0f, 3.0f, 4.0f};

		std::cout << "model: " << model(in) << std::endl;

		Gradient grad = model.gradient({in}, {{1, 1}});

		std::cout << "gradients:\n";
		for (auto &g : grad.grads)
			std::cout << "\t" << g << std::endl;

		std::cout << "igrads:\n";
		for (auto &g : grad.igrads)
			std::cout << "\t" << g << std::endl;
	}

	detail::MemoryTracker::report();
}
