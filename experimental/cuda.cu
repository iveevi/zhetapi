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

		std::cout << "a:" << a << " => " << a.verbose() << std::endl;
		std::cout << "b:" << b << " => " << b.verbose() << std::endl;

		Tensor <float> ::set_variant(eCUDA);

		Tensor <float> c = a + b;

		std::cout << "c: " << c << " => " << c.verbose() << std::endl;

		// jitify and fill memory as such...
		Tensor <float> d = Tensor <float> (Tensor <float> ::shape_type {2, 2});
		std::cout << "d:" << d << " => " << d.verbose() << std::endl;
		
		Tensor <float> e = Tensor <float> (Tensor <float> ::shape_type {2, 2});
		std::cout << "e:" << d << " => " << d.verbose() << std::endl;

		c.copy(a);
		c.copy(d);

		d.copy(a);
		e.copy(d);
		
		std::cout << "\nc:" << c << " => " << c.verbose() << std::endl;
		std::cout << "d:" << d << " => " << d.verbose() << std::endl;
		std::cout << "e:" << e << " => " << e.verbose() << std::endl;

		// TODO: manual array copy
		
		detail::MemoryTracker::report();
	}

	detail::MemoryTracker::report();
}
