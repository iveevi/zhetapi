#include "../include/autograd/autograd.hpp"
#include "../include/io/print.hpp"
#include "../include/range.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

int main()
{
	/* TODO: test in testing dir
	Variable x, y;

	// Differentiation test
	// Function k = x/y;
	Function k = autograd::sqrt(x + x);
	Function h = k.differentiate(0);

	// TODO: input size exception
	std::cout << "h = " << h(2.0) << std::endl;
	std::cout << "h = " << h(4.0) << std::endl; */

	auto r = Range <float> (0, 10, 0.1);
	std::cout << "r.size() = " << r.size() << std::endl;
	std::cout << "r.length() = " << r.length() << std::endl;
	std::cout << "r = " << r << std::endl;

	Constant c(
		{3, 3, 3},
		[](size_t i) {
			return i;
		}
	);

	std::cout << "c = " << c << std::endl;
	std::cout << "c.shape = (";
	for (auto i : c.shape())
		std::cout << i << ", ";
	std::cout << ")" << std::endl;
	std::cout << "c.size = " << c.size() << std::endl;
	std::cout << "c[9] = " << c.get(9) << std::endl;

	Constant slice = c[{0, 3, 2}];

	std::cout << "\nslice = " << slice << std::endl;
	std::cout << "slice.shape = (";
	for (auto i : slice.shape())
		std::cout << i << ", ";
	std::cout << ")" << std::endl;
	std::cout << "slice.size = " << slice.size() << std::endl;
	std::cout << "slice[9] = " << slice.get(9) << std::endl;

	Constant::slice_type s = {0, 3, 2};

	std::cout << "\n2nd index of slice = " << s(2) << std::endl;
}
