#include <iostream>

#include "include/tensor.hpp"

using namespace zhetapi;

struct CuTensor {
	size_t dimensions;
	size_t *shape;
	float *array; // Borrows data from Tensor
};

struct CuMatrix {
	size_t rows;
	size_t columns;
	float *array; // Borrows data from Tensor
};

int main()
{
	{
		Tensor <float> a = Tensor <float> ::ones({2, 2});
		Tensor <float> b = Tensor <float> ::zeros({2, 2});

		std::cout << a << std::endl;
		std::cout << b << std::endl;

		Tensor <float> ::set_variant(eCUDA);

		Tensor <float> c = a + b;

		std::cout << c << std::endl;

		// jitify and fill memory as such...
		// Tensor <float> d = Tensor <float> ::ones({2, 2});
		
		detail::MemoryTracker::report();
	}

	detail::MemoryTracker::report();
}
