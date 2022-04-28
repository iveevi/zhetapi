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

	Constant slice1 = c[{0, 3, 2}];

	std::cout << "\nslice1 = " << slice1 << std::endl;
	std::cout << "slice1.shape = (";
	for (auto i : slice1.shape())
		std::cout << i << ", ";
	std::cout << ")" << std::endl;
	std::cout << "slice1.size = " << slice1.size() << std::endl;
	std::cout << "slice1[9] = " << slice1.get(9) << std::endl;

	Constant slice2 = c[{0, 3, 2}][{0, 3, 2}];
	std::cout << "\nslice2 = " << slice2 << std::endl;
	std::cout << "slice2.shape = (";
	for (auto i : slice2.shape())
		std::cout << i << ", ";
	std::cout << ")" << std::endl;
	std::cout << "slice2.size = " << slice2.size() << std::endl;
	std::cout << "slice2[8] = " << slice2.get(8) << std::endl;

	Constant index1 = c[1];
	std::cout << "\nindex1 = " << index1 << std::endl;
	std::cout << "index1.shape = (";
	for (auto i : index1.shape())
		std::cout << i << ", ";
	std::cout << ")" << std::endl;
	std::cout << "index1.size = " << index1.size() << std::endl;
	std::cout << "index1[9] = " << index1.get(9) << std::endl;

	Constant index2 = c[1][1];
	std::cout << "\nindex2 = " << index2 << std::endl;
	std::cout << "index2.shape = (";
	for (auto i : index2.shape())
		std::cout << i << ", ";
	std::cout << ")" << std::endl;

	std::cout << "index3 = " << index2[1] << ", size = " << index2.size() << std::endl;
}
