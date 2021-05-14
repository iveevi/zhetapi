#ifndef PORT_H_
#define PORT_H_

// C/C++ headers
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>

#include <ctime>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

// Engine headers
#include <all/zhplib.hpp>

#include <vector.hpp>
#include <matrix.hpp>
#include <tensor.hpp>
#include <fourier.hpp>
#include <polynomial.hpp>

#include <std/calculus.hpp>
#include <std/functions.hpp>
#include <std/interval.hpp>
#include <std/linalg.hpp>
#include <std/activations.hpp>
#include <std/erfs.hpp>

// Macros
#define TEST(name)	bool name(ostringstream &oss)
#define RIG(name)	{#name, &name}	

// Namespaces
using namespace std;

// Typedefs
using tclk = chrono::high_resolution_clock;
using tpoint = chrono::high_resolution_clock::time_point;

// Timers
extern tclk clk;
extern tpoint tmp;

// Bench marking structures
struct bench {
	tpoint epoch;

	bench() : epoch(clk.now()) {}
	bench(const tpoint &t) : epoch(t) {}
};

ostream &operator<<(ostream &, const bench &);

// Test functions
TEST(gamma_and_factorial);

TEST(vector_construction_and_memory);
TEST(vector_operations);

TEST(matrix_construction_and_memory);
TEST(tensor_construction_and_memory);

TEST(integration);

TEST(function_computation);

TEST(interval_construction);
TEST(interval_sampling);

TEST(diag_matrix);
TEST(qr_decomp);
TEST(lq_decomp);
TEST(qr_alg);
TEST(matrix_props);

TEST(fourier_series);

TEST(polynomial_construction);
TEST(polynomial_comparison);
TEST(polynomial_arithmetic);

TEST(act_linear);
TEST(act_relu);

#endif
