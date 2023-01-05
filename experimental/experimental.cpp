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

		auto model = ml::dense(10, 10);
		auto y = model(x);
		model.fn("dropout", 0.5f);

		std::cout << model.summary() << std::endl;
		std::cout << y.summary() << std::endl;
	}

	detail::MemoryTracker::report();
}
