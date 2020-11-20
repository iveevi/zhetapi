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
#include <function.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <tensor.hpp>

using namespace std;

// Bench marking structures
struct bench {};

extern bench mark;

ostream &operator<<(ostream &, const bench &);

// Timers
extern chrono::high_resolution_clock clk;

extern chrono::high_resolution_clock::time_point epoch;
extern chrono::high_resolution_clock::time_point tmp;

// Test functions
bool vector_construction_and_memory();
bool matrix_construction_and_memory();
bool tensor_construction_and_memory();

bool function_compilation_testing();

#endif