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
#include "../../engine/all/zhplib.hpp"

#include "../../engine/fourier.hpp"
#include "../../engine/linalg.hpp"
#include "../../engine/matrix.hpp"
#include "../../engine/module.hpp"
#include "../../engine/polynomial.hpp"
#include "../../engine/tensor.hpp"
#include "../../engine/token.hpp"
#include "../../engine/vector.hpp"

#include "../../engine/std/calculus.hpp"
#include "../../engine/std/functions.hpp"
#include "../../engine/std/interval.hpp"
#include "../../engine/std/activations.hpp"
#include "../../engine/std/erfs.hpp"

#include "../../engine/lang/parser.hpp"
#include "../../engine/lang/feeder.hpp"

#include "../../engine/core/node_manager.hpp"
#include "../../engine/core/raw_types.hpp"
#include "../../engine/core/collection.hpp"

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
TEST(act_leaky_relu);
TEST(act_sigmoid);

TEST(module_construction);

TEST(parsing_global_assignment);
TEST(parsing_global_branching);

TEST(compile_operand);
TEST(compile_const_exprs);
TEST(compile_var_exprs);

#endif
