#include "../include/autograd/autograd.hpp"
#include "../include/io/print.hpp"
// #include "../include/common.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

template <size_t T, size_t ... Args>
std::vector <size_t> collect()
{
	std::vector <size_t> pvec = collect <Args...> ();
	pvec.push_back(T);
	return pvec;
}

template <size_t T>
std::vector <size_t> collect()
{
	return std::vector <size_t> {T};
}

int main()
{
	// TODO: test in testing dir
	// TODO: benchmark program
	Variable x, y;

	// Differentiation test
	// Function k = x/y;
	Function k = autograd::sqrt(x + x);
	Function h = k.differentiate(0);

	// TODO: input size exception
	std::cout << "k = " << k(2.0) << ", h = " << h(2.0) << std::endl;
	std::cout << "k = " << k(4.0) << ", h = " << h(4.0) << std::endl;

	auto vec = collect <4, 4> ();
}