#include "include/autograd/ml.hpp"
#include "include/autograd/activation.hpp"
#include "include/autograd/gradient_queue.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	{
		Variable x;
		auto model = ml::dense(3, 3);

		Constant input {
			Constant::shape_type {3, 3, 3},
			1.0f
		};

		std::cout << "input = " << input.verbose() << "\n";
		std::cout << "model(input): = " << model(input).verbose() << "\n";

		Constant igrad { Constant::shape_type {3, 3, 3}, 1.0f };
		Gradient grads = model.gradient({input}, {igrad});

		std::cout << "grads = " << grads.igrads[0].verbose() << "\n";
	}

	detail::MemoryTracker::report();
}
