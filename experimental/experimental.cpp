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
	// Target value
	utility::Interval<> rng(0, 1);
	Constant x1 {2.0, 2.0, 2.0, 2.0, 2.0};
	Constant x2 {1, 2, 3, 4, 5};
	Constant x3 {{5}, [&](size_t) {return rng();}};

	std::cout << "softmax(x1) = " << ml::softmax(x1) << std::endl;
	std::cout << "\tigrad = " << ml::softmax.gradient({x1}).igrads[0]/x1 << std::endl;

	std::cout << "softmax(x2) = " << ml::softmax(x2) << std::endl;
	std::cout << "\tigrad = " << ml::softmax.gradient({x2}).igrads[0]/x2 << std::endl;

	// std::cout << "Loop:\n";

	Variable x, y;
	auto mse = square(length(x - y))/Constant(5);
	auto true_dmse = Constant(2) * (x - y)/Constant(5);

	Constant target {0.1, 0.2, 0.2, 0.3, 0.2};

	// TODO: minus operator
	auto cross_entropy = -1.0f * autograd::dot(autograd::log(x), y);
	auto true_dcross_entropy = (y/x).refactored(x, y);

	std::cout << "true dce:\n";
	std::cout << true_dcross_entropy.summary() << std::endl;
	std::cout << true_dcross_entropy(1, 0) << std::endl;

	/* for (int i = 0; i < 100; i++) {
		std::cout << "\nx3 = " << x3 << std::endl;

		auto o = ml::softmax(x3);
		std::cout << "softmax(x3) = " << o << std::endl;
		std::cout << "errf = " << mse(o, target) << std::endl;
		std::cout << "cross = " << cross_entropy(o, target) << std::endl;

		auto igrad = true_dcross_entropy(o.flat(), target);
		std::cout << "\tigrad = " << igrad << std::endl;

		x3 -= 0.1f * igrad;
	} */
}
