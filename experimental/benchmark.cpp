// Standard headers
#include <iostream>
#include <chrono>

// Library headers
#include "../include/autograd/autograd.hpp"

// Number of iterations
constexpr int N = 1000000;

int main()
{
	using namespace zhetapi;

	// Clocking tools
	std::chrono::high_resolution_clock::time_point start, end;

	// Using standard and autograd
	auto f_standard = [](const autograd::Constant &x) {
		autograd::Constant sqrt = x.transform(
			[](long double x) {
				return std::sqrt(x);
			}
		);

		autograd::Constant sin = x.transform(
			[](long double x) {
				return std::sin(x);
			}
		);

		return multiply(x, sqrt) + sin - x;
	};

	autograd::Variable x;
	autograd::Function f_autograd = x * autograd::sqrt(x)
		+ autograd::sin(x) - x;

	// Input
	autograd::Constant input = {
		// Dimensions
		{3, 3, 3},
		// Values
		{
			1.0, 2.0, 3.0,
			4.0, 5.0, 6.0,
			7.0, 8.0, 9.0,

			1.0, 2.0, 3.0,
			4.0, 5.0, 6.0,
			7.0, 8.0, 9.0,

			1.0, 2.0, 3.0,
			4.0, 5.0, 6.0,
			7.0, 8.0, 9.0
		}
	};

	// Check that the outputs are the same
	std::cout << "Preliminary check:\n";
	std::cout << "\tStandard: " << f_standard(input) << std::endl;
	std::cout << "\tAutograd: " << f_autograd(input) << std::endl;

	autograd::Constant out_standard = f_standard(input);
	autograd::Constant out_autograd = f_autograd(input);

	if (out_standard != out_autograd) {
		std::cerr << "Outputs are not the same!\n";
		return 1;
	}

	std::cout << "\nCheck passed!\n";

	// Benchmarking
	std::cout << "\nBenchmarking..." << std::endl;

	// Standard
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < N; i++)
		f_standard(input);

	end = std::chrono::high_resolution_clock::now();

	auto t1 = std::chrono::duration_cast
		<std::chrono::microseconds> (end - start).count();

	// Autograd
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < N; i++)
		f_autograd(input);

	end = std::chrono::high_resolution_clock::now();

	auto t2 = std::chrono::duration_cast
		<std::chrono::microseconds> (end - start).count();
	
	// Summary
	std::cout << "Standard: " << (double) t1 / 1000000.0
		<< " seconds (" << t1 << " microseconds)" << std::endl;

	std::cout << "Autograd: " << (double) t2 / 1000000.0
		<< " seconds (" << t2 << " microseconds)" << std::endl;

	std::cout << "Speedup: " << (100.0 * (double) t2 / t1)
		<< "%" << std::endl;
}
