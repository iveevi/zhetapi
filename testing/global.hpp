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
#include "../zhetapi.hpp"

// Macros
#define TEST(name)	bool name(ostringstream &oss, int cout)
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

// Coloring
struct term_colors {
	string color;
};

extern term_colors reset;

extern term_colors bred;
extern term_colors bgreen;
extern term_colors byellow;

ostream &operator<<(ostream &, const term_colors &);

struct term_ok {};
struct term_err {};

extern term_ok ok;
extern term_err err;

ostream &operator<<(ostream &, const term_ok &);
ostream &operator<<(ostream &, const term_err &);

// Test functions
TEST(gamma_and_factorial);

TEST(vector_construction_and_memory);
TEST(vector_operations);

TEST(matrix_construction_and_memory);
TEST(kernel_apt_and_mult);
TEST(kernel_rmt_and_mult);
TEST(kernel_vvt_mult);

TEST(tensor_construction_and_memory);

TEST(integration);

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
TEST(act_leaky_relu);
TEST(act_sigmoid);

#endif
