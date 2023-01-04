#include "../include/autograd/ml.hpp"
#include "../include/autograd/activation.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

#include <variant>
#include <unordered_map>
#include <functional>

int main()
{
	{
		Variable x;
		Variable y;
	
		// auto loss = square(length(x - y))/Constant {10};
		auto dloss = 2 * (x - y)/Constant {10};
		std::cout << "dloss(10, 1) = " << dloss(10, 1) << std::endl;
	
		/* auto model = ml::dense(100, 30)(x);
		model = ml::sigmoid(model);
		model = ml::dense(30, 10)(model);
		model = ml::softmax(model); */

		detail::MemoryTracker::report();
	}

	detail::MemoryTracker::report();
}
