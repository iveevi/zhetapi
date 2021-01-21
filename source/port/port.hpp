#ifndef PORT_H_
#define PORT_H_

// C/C++ headers
#include <chrono>
#include <iostream>
#include <vector>

#include <ctime>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

// Engine headers
#include <zhplib.hpp>

#include <vector.hpp>
#include <matrix.hpp>
#include <tensor.hpp>

#include <std/functions.hpp>

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
bool gamma_and_factorial();

bool vector_construction_and_memory();
bool matrix_construction_and_memory();
bool tensor_construction_and_memory();

bool function_computation();
bool function_compilation_testing();

#endif
